import asyncio
import os
import time
from collections import Counter

from PyQt5.QtCore import pyqtSignal, QThread

from config import Config

from helpers.GameHelper import GameHelper, AnimationArea
from helpers.ImageLocator import ImageLocator
from helpers.ScreenHelper import ScreenHelper
from helpers.WechatCardRecognizer import WechatCardRecognizer

from models import BidModel
from models import FarmerModel
from models import LandlordModel

from douzero.env.game_new import GameEnv
from douzero.env.move_detector import get_move_type
from douzero.evaluation.deep_agent_new import DeepAgent

from constants import RealCard2EnvCard, EnvCard2RealCard, AllEnvCard, AutomaticModeEnum
from utils import remove_chars_from_string
from inference import HandInferenceEngine

# 玩家位置（0：地主上家，1：地主，2：地主下家）
PlayerPosition = ['landlord_up', 'landlord', 'landlord_down']

class WorkerThread(QThread):
    card_recorder_signal = pyqtSignal(str)
    three_cards_signal = pyqtSignal(str)
    my_position_signal = pyqtSignal(str)
    ai_suggestion_signal = pyqtSignal(list)
    bid_win_rate_signal = pyqtSignal(list)
    game_win_rate_signal = pyqtSignal(float)
    played_card_signal = pyqtSignal(list)
    hand_inference_signal = pyqtSignal(dict)
    remaining_count_signal = pyqtSignal(dict)

    def __init__(self, automatic_mode, bid_threshold, redouble_threshold, super_redouble_threshold, mingpai_threshold):
        super(WorkerThread, self).__init__()
        self.config = Config.load()
        self.screenHelper = ScreenHelper()
        self.imageLocator = ImageLocator(self.screenHelper)
        self.gameHelper = GameHelper(self.imageLocator, self.screenHelper)
        self.wechatRecognizer = WechatCardRecognizer() if getattr(self.config, 'platform', '') == 'wechat_miniapp' else None
        
        self.bid_threshold = bid_threshold if bid_threshold else self.config.bid_threshold
        self.redouble_threshold = redouble_threshold if redouble_threshold else self.config.redouble_threshold
        self.super_redouble_threshold = super_redouble_threshold if super_redouble_threshold else self.config.super_redouble_threshold
        self.mingpai_threshold = mingpai_threshold if mingpai_threshold else self.config.mingpai_threshold

        print('self.bid_threshold: ', self.bid_threshold)
        print('self.redouble_threshold: ', self.redouble_threshold)
        print('self.super_redouble_threshold: ', self.super_redouble_threshold)
        print('self.mingpai_threshold: ', self.mingpai_threshold)

        self.automatic_mode = automatic_mode
        self.worker_runing = False              # 线程是否在运行

        # self.auto_play_cards = False            # 是否自动打牌（无需人为操作）
        self.in_game_start_screen = False       # 是否进入开始游戏界面
        self.game_started = False               # 游戏是否已开局
        self.landlord_confirmed = False         # 是否已确认地主
        
        self.three_cards = ''                   # 三张底牌
        self.my_hand_cards = ''                 # 我的手牌

        self.my_position_code = None            # 我的位置（角色）代码（0：landlord_up, 1：landlord, 2：landlord_down）
        self.my_position = None                 # 我的位置（角色）

        self.data_initializing = False          # 是否正在初始化数据（执行 initial_data 函数）
        self.data_initialized = False           # 是否完成数据初始化

        self.play_order = None                  # 出牌顺序（0：我先出牌，1：我的下家先出牌，2：我的上家先出牌）
        self.play_order_of_next = None          # 下一次出牌顺序（0：我，1：我的下家，2：我的上家）

        self.card_playing = False               # 是否在出牌中（已经有人出过牌为 true 否则 false）
        
        self.other_hands_cards = []             # 其他玩家的手牌（整副牌减去我的手牌，后续再减掉历史出牌）
        self.other_hands_cards_str = ''
        self.all_player_card_data = {}

        self.env = None
        self.three_cards_env = None
        self.my_hand_cards_env = None
        self.my_played_cards_env = None
        self.other_played_cards_env = None
        
        self.waiting_for_animation_to_end = False   # 是否正在等待动画结束
        self.right_played_completed = False         # 右侧玩家是否完成一次出牌
        self.left_played_completed = False          # 左侧玩家是否完成一次出牌
        self.my_played_completed = False            # 我是否完成一次出牌

        self.in_bidding_progress = False
        self.in_redouble_progress = False

        self.player_bidding_status = None

        self.ai_suggested_received = False
        self.my_played_card_clicked = False

        self.action_message = None
        self.action_list = None

        # 概率推牌器：利用已知手牌、底牌、出牌和 Pass 动作推测两家剩余手牌。
        self.hand_inference = None

        self.try_num = 3
        self.round_count = 0

        self.model_path_dict = {
            'landlord': "baselines/resnet/resnet_landlord.ckpt",
            'landlord_up': "baselines/resnet/resnet_landlord_up.ckpt",
            'landlord_down': "baselines/resnet/resnet_landlord_down.ckpt"
        }

        LandlordModel.init_model("baselines/resnet/resnet_landlord.ckpt")

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.run_task())
        finally:
            self.loop.close()

    async def run_task(self):
        self.worker_runing = True
        self.player_bidding_status: dict[int, list[int]] = {}

        if getattr(self.config, 'platform', '') == 'wechat_miniapp':
            await self.run_wechat_probe()
            return

        while self.worker_runing:
            print()
            print("----- WORKER STARTED -----")
            print()

            # 第一次截图，用于获取窗口宽高等数据
            await self.screenHelper.getScreenshot()

            # 检测是否开局
            while self.worker_runing and not self.game_started:
                await self.before_start()
                time.sleep(1)

            # 确认地主之后，并且数据尚未初始化
            while self.worker_runing and self.landlord_confirmed and not self.data_initialized:
                # 如果`正在初始化中`或`已初始化完成`，就直接 continue 跳过本次循环
                if self.data_initializing or self.data_initialized:
                    continue

                await self.initial_data()
            
            # 初始化数据之后，并且出牌尚未开始
            while self.worker_runing and self.data_initialized and not self.card_playing:
                await self.run_game()

            await asyncio.sleep(1)
            
            print()
            print("----- WORKER FINISHED -----")
            print()

    async def run_wechat_probe(self):
        """Run the WeChat miniapp as a read-only live state/inference pipeline.

        The three user-recorded games are used as calibration evidence:
        - hand changes are authoritative for the local player;
        - opponent remaining-count drops trigger opponent plays;
        - the visible play must contain exactly the number of cards that left
          the opponent's hand before the action is committed;
        - if a later confirmed actor appears, skipped seats are logically Pass.
        """
        print("微信小程序牌局接入模式已启动")
        print("三盘录像标定版：手牌差分 + 剩余张数触发 + 出牌张数交叉校验")
        print("当前仍为只读模式，不会自动点击游戏")
        print()

        recognizer = self.wechatRecognizer or WechatCardRecognizer()
        screenshot_saved = False
        missing_reported = False
        pending = {}

        round_initialized = False
        round_signature = None
        expected_side = None
        confirmed_my_hand = None
        pre_landlord_hand = None
        preinit_counts = {"left": None, "right": None}
        tracked_remaining = {"left": None, "right": None}
        observed_remaining = {"left": None, "right": None}
        count_desync = {"left": False, "right": False}
        recent_play = {
            "left": {"cards": "", "time": 0.0},
            "right": {"cards": "", "time": 0.0},
        }
        count_missing_frames = {"left": 0, "right": 0}
        self_hand_missing_frames = 0
        pass_latched = {"left": False, "right": False, "me": False}
        last_state = {
            "three_cards": None,
            "position_code": None,
        }
        wechat_other_hands_cards_str = ""

        side_cycle = {"me": "right", "right": "left", "left": "me"}
        landlord_start_side = {0: "right", 1: "me", 2: "left"}

        def display_cards(cards):
            if not cards:
                return "-"
            display_map = {"D": "大王", "X": "小王", "T": "10"}
            return " ".join(display_map.get(card, card) for card in cards)

        def hand_difference(before, after):
            """Return cards removed from before to after; None if after is impossible."""
            before_counter = Counter(before)
            after_counter = Counter(after)
            if any(after_counter[card] > before_counter[card] for card in after_counter):
                return None

            missing = before_counter - after_counter
            result = []
            for card in before:
                if missing[card] > 0:
                    result.append(card)
                    missing[card] -= 1
            return "".join(result)

        def stable_value(key, value, frames=2):
            previous, count = pending.get(key, (None, 0))
            if value == previous:
                count += 1
            else:
                previous, count = value, 1
            pending[key] = (previous, count)
            return value if count >= frames else None

        def player_for_side(side):
            if self.my_position_code is None:
                return None
            if side == "me":
                return self.my_position
            if side == "right":
                return PlayerPosition[(self.my_position_code + 1) % 3]
            if side == "left":
                return PlayerPosition[(self.my_position_code + 2) % 3]
            return None

        def emit_remaining_counts():
            result = {}
            for side in ("left", "right"):
                player = player_for_side(side)
                value = observed_remaining[side]
                if value is None:
                    value = tracked_remaining[side]
                if player is not None and value is not None:
                    result[player] = {
                        "count": int(value),
                        "observed": observed_remaining[side] is not None,
                        "desync": bool(count_desync[side]),
                    }
            if result:
                self.remaining_count_signal.emit(result)

        def record_pass(side, inferred=False):
            nonlocal expected_side
            player = player_for_side(side)
            if player is None:
                return False
            label = {"left": "左侧玩家", "right": "右侧玩家", "me": "我"}[side]
            suffix = " [由后续行动反推]" if inferred else ""
            print(f"微信牌局 >>> {label}：不出{suffix}")
            self.played_card_signal.emit([player, "Pass"])
            self.record_hand_inference_action(player, "")
            apply_action_to_douzero(player, "")
            expected_side = side_cycle[side]
            return True

        def emit_douzero_suggestion_if_my_turn():
            if self.env is None or self.env.game_over:
                self.ai_suggestion_signal.emit([])
                return
            if self.env.acting_player_position != self.my_position:
                self.ai_suggestion_signal.emit([])
                return

            try:
                action_message, action_list = self.env.step(
                    self.my_position, action=None, update=False
                )
                self.action_message = action_message
                self.action_list = action_list[:3]
                self.ai_suggestion_signal.emit(self.action_list)

                action_text = action_message.get("action", "")
                if action_text:
                    print(
                        f"DouZero建议 >>> {display_cards(action_text)} "
                        f"(评分 {action_message.get('win_rate', 0):.3f})"
                    )
                else:
                    print("DouZero建议 >>> 不出")
            except Exception as exc:
                self.ai_suggestion_signal.emit([])
                print(f"DouZero建议计算失败（不影响识牌/推牌）: {exc}")

        def apply_action_to_douzero(player, cards):
            if self.env is None:
                return False
            if self.env.game_over:
                self.ai_suggestion_signal.emit([])
                return False

            if self.env.acting_player_position != player:
                print(
                    "DouZero状态不同步 >>> "
                    f"环境等待 {self.env.acting_player_position}，"
                    f"识别到 {player}；本次不写入AI环境"
                )
                self.ai_suggestion_signal.emit([])
                return False

            try:
                action_env = sorted(
                    [RealCard2EnvCard[card] for card in cards]
                )
                self.env.step(player, action=action_env, update=True)
                emit_douzero_suggestion_if_my_turn()
                return True
            except Exception as exc:
                self.ai_suggestion_signal.emit([])
                print(f"DouZero状态更新失败（不影响识牌/推牌）: {exc}")
                return False

        def initialize_round(my_hand, three_cards, position_code):
            nonlocal round_initialized, round_signature, expected_side
            nonlocal confirmed_my_hand, wechat_other_hands_cards_str

            self.my_hand_cards = my_hand
            self.three_cards = three_cards
            self.my_position_code = position_code
            self.my_position = PlayerPosition[position_code]
            confirmed_my_hand = my_hand

            self.hand_inference = HandInferenceEngine(
                my_position=self.my_position,
                my_hand_cards=my_hand,
                three_landlord_cards=three_cards,
            )

            # Build the original DouZero environment from the same confirmed
            # WeChat state. Opponent card identities are intentionally unknown;
            # game_new.py tracks their counts while the user's infoset uses the
            # complete unseen-card pool, matching the original project design.
            self.my_hand_cards_env = sorted(
                [RealCard2EnvCard[card] for card in my_hand]
            )
            self.three_cards_env = sorted(
                [RealCard2EnvCard[card] for card in three_cards]
            )
            self.other_hands_cards = []
            self.all_player_card_data = {}
            self.env = None
            try:
                self.initOtherPlayerHandCards()
                self.initAllPlayerCardData()
                self.create_ai_representer()
                self.env.card_play_init(self.all_player_card_data)
                print("DouZero只读建议环境已初始化")
            except Exception as exc:
                self.env = None
                self.ai_suggestion_signal.emit([])
                print(f"DouZero初始化失败（识牌/推牌继续运行）: {exc}")

            landlord_side = landlord_start_side[position_code]
            for side in ("left", "right"):
                tracked_remaining[side] = 20 if side == landlord_side else 17
                observed_remaining[side] = (
                    preinit_counts[side]
                    if preinit_counts[side] == tracked_remaining[side]
                    else None
                )
                count_desync[side] = False

            if self.other_hands_cards_str:
                wechat_other_hands_cards_str = self.other_hands_cards_str
            else:
                remaining = list(AllEnvCard)
                for card in [RealCard2EnvCard[ch] for ch in my_hand]:
                    if card in remaining:
                        remaining.remove(card)
                wechat_other_hands_cards_str = ''.join(
                    [EnvCard2RealCard[x] for x in remaining]
                )[::-1]
                self.other_hands_cards_str = wechat_other_hands_cards_str

            self.card_recorder_signal.emit(wechat_other_hands_cards_str)
            self.three_cards_signal.emit(three_cards)
            self.my_position_signal.emit(self.my_position)
            emit_remaining_counts()

            expected_side = landlord_start_side[position_code]
            round_signature = (my_hand, three_cards, position_code)
            round_initialized = True
            for side in pass_latched:
                pass_latched[side] = False
            for side in recent_play:
                recent_play[side] = {"cards": "", "time": 0.0}

            print()
            print("===== 微信牌局状态已初始化 =====")
            print(f"我的初始手牌({len(my_hand)}): {display_cards(my_hand)}")
            print(f"三张底牌: {display_cards(three_cards)}")
            print(f"我的身份: {self.my_position}")
            print(f"首个行动方: {expected_side}")
            print("==============================")
            print()
            self.refresh_hand_inference()
            emit_douzero_suggestion_if_my_turn()

        def pending_self_change(raw_hand):
            if not round_initialized or not confirmed_my_hand or not raw_hand:
                return False
            removed = hand_difference(confirmed_my_hand, raw_hand)
            return removed is not None and len(removed) > 0

        def can_infer_pass(side, live_counts, raw_hand):
            if side == "me":
                return not pending_self_change(raw_hand)
            count = live_counts.get(side)
            tracked = tracked_remaining.get(side)
            if count is not None and tracked is not None and count < tracked:
                return False
            return True

        def sync_to_actor(actor, live_counts, raw_hand):
            nonlocal expected_side
            guard = 0
            while expected_side is not None and expected_side != actor and guard < 3:
                if not can_infer_pass(expected_side, live_counts, raw_hand):
                    return False
                if not record_pass(expected_side, inferred=True):
                    return False
                guard += 1
            return expected_side == actor

        while self.worker_runing:
            screenshot, _ = await self.screenHelper.getScreenshot()
            if screenshot is None:
                if not missing_reported:
                    print("未找到或无法截图‘腾讯欢乐斗地主’窗口，请保持小程序窗口打开")
                    missing_reported = True
                await asyncio.sleep(0.5)
                continue

            missing_reported = False
            if not screenshot_saved:
                os.makedirs('screenshots', exist_ok=True)
                calibration_path = os.path.join('screenshots', 'wechat_calibration.png')
                screenshot.save(calibration_path)
                description = self.screenHelper.get_window_description() or {}
                print(
                    f"已连接微信斗地主窗口："
                    f"{description.get('title', '腾讯欢乐斗地主')} "
                    f"[{description.get('class_name', '-')} ]"
                )
                print(f"客户区截图尺寸：{screenshot.size[0]} x {screenshot.size[1]}")
                print(f"标定截图已保存：{calibration_path}")
                print()
                screenshot_saved = True

            try:
                now = time.monotonic()
                raw_hand = recognizer.recognize_my_hand(screenshot)
                init_hand = stable_value("init_hand", raw_hand, frames=5)
                live_hand = stable_value("live_hand", raw_hand, frames=3)

                raw_counts = {
                    side: recognizer.recognize_remaining_count(screenshot, side, expected=None)
                    for side in ("left", "right")
                }
                for side in ("left", "right"):
                    if raw_counts[side] is None:
                        count_missing_frames[side] += 1
                    else:
                        count_missing_frames[side] = 0

                if round_initialized and not raw_hand:
                    self_hand_missing_frames += 1
                else:
                    self_hand_missing_frames = 0
                live_counts = {
                    side: stable_value(f"live_{side}_count", raw_counts[side], frames=4)
                    for side in ("left", "right")
                }

                if not round_initialized:
                    for side in ("left", "right"):
                        if live_counts[side] is not None:
                            preinit_counts[side] = live_counts[side]

                    if init_hand is not None and len(init_hand) == 17:
                        pre_landlord_hand = init_hand

                    landlord_side = recognizer.detect_landlord_side(screenshot)
                    position_map = {"right": 0, "me": 1, "left": 2}
                    badge_position = stable_value(
                        "badge_position", position_map.get(landlord_side), frames=3
                    )
                    inferred_position = badge_position
                    if init_hand is not None and len(init_hand) == 20:
                        inferred_position = 1
                    elif preinit_counts["left"] == 20:
                        inferred_position = 2
                    elif preinit_counts["right"] == 20:
                        inferred_position = 0

                    position_code = stable_value(
                        "position_code", inferred_position, frames=3
                    )
                    if (
                        position_code is not None
                        and position_code != last_state["position_code"]
                    ):
                        position_text = {
                            0: "农民（地主上家）",
                            1: "地主",
                            2: "农民（地主下家）",
                        }[position_code]
                        evidence = (
                            "20张手牌" if position_code == 1
                            else "左侧20张" if position_code == 2 and preinit_counts["left"] == 20
                            else "右侧20张" if position_code == 0 and preinit_counts["right"] == 20
                            else "地主标志"
                        )
                        print(f"微信专用识牌 >>> 我的身份: {position_text} [{evidence}]")
                        last_state["position_code"] = position_code

                    top_bottom = stable_value(
                        "three_cards", recognizer.recognize_bottom_cards(screenshot), frames=2
                    )
                    three_cards = top_bottom
                    if (
                        last_state["position_code"] == 1
                        and init_hand is not None
                        and len(init_hand) == 20
                        and pre_landlord_hand is not None
                    ):
                        added = hand_difference(init_hand, pre_landlord_hand)
                        if added is not None and len(added) == 3:
                            three_cards = added

                    if three_cards is not None and len(three_cards) == 3:
                        if three_cards != last_state["three_cards"]:
                            source = (
                                " [17→20手牌差分]"
                                if last_state["position_code"] == 1
                                else ""
                            )
                            print(
                                f"微信专用识牌 >>> 三张底牌: "
                                f"{display_cards(three_cards)}{source}"
                            )
                            last_state["three_cards"] = three_cards

                    ready_position = last_state["position_code"]
                    ready_hand = init_hand or ""
                    ready_three = last_state["three_cards"] or ""
                    expected_count = (
                        20 if ready_position == 1
                        else 17 if ready_position is not None
                        else None
                    )
                    if (
                        expected_count is not None
                        and len(ready_hand) == expected_count
                        and len(ready_three) == 3
                    ):
                        signature = (ready_hand, ready_three, ready_position)
                        if signature != round_signature:
                            initialize_round(ready_hand, ready_three, ready_position)

                if not round_initialized:
                    await asyncio.sleep(0.35)
                    continue

                # Keep stable play candidates for both opponents, independent of
                # whose turn we currently think it is.  Count changes decide
                # whether the candidate becomes a committed action.
                played_values = {
                    "left": recognizer.recognize_left_played(screenshot),
                    "right": recognizer.recognize_right_played(screenshot),
                    "me": recognizer.recognize_my_played(screenshot),
                }
                for side in ("left", "right"):
                    cards = stable_value(
                        f"{side}_play_candidate", played_values[side], frames=2
                    )
                    if cards:
                        recent_play[side] = {"cards": cards, "time": now}

                # Prefer an action from the currently expected seat.  If a later
                # seat has hard evidence, skipped seats are logically Pass.
                self_removed = None
                self_final_out = False
                if live_hand and confirmed_my_hand and live_hand != confirmed_my_hand:
                    self_removed = hand_difference(confirmed_my_hand, live_hand)
                    if self_removed == "":
                        confirmed_my_hand = live_hand
                        self_removed = None

                if (
                    not self_removed
                    and confirmed_my_hand
                    and self_hand_missing_frames >= 4
                ):
                    final_cards = recognizer.recognize_my_played(
                        screenshot, expected_count=len(confirmed_my_hand)
                    )
                    if final_cards and len(final_cards) == len(confirmed_my_hand):
                        self_removed = final_cards
                        self_final_out = True

                def opponent_candidate(side):
                    new_count = live_counts.get(side)
                    old_count = tracked_remaining.get(side)
                    if old_count is None:
                        return None

                    drop = None
                    target_count = None
                    source = None
                    if new_count is not None and new_count < old_count:
                        drop = old_count - new_count
                        target_count = new_count
                        source = "剩余张数下降"
                    elif (
                        new_count is None
                        and count_missing_frames[side] >= 4
                        and old_count > 0
                    ):
                        # On the final play the blue count badge disappears
                        # immediately instead of showing 0.  If the visible
                        # play contains every remaining card, that is hard
                        # evidence that the player went out.
                        drop = old_count
                        target_count = 0
                        source = "牌数框消失+全手出完"
                    else:
                        return None

                    info = recent_play[side]
                    cards = info["cards"] if now - info["time"] <= 3.0 else ""
                    if not cards or len(cards) != drop:
                        if side == "left":
                            cards = recognizer.recognize_left_played(
                                screenshot, expected_count=drop
                            )
                        else:
                            cards = recognizer.recognize_right_played(
                                screenshot, expected_count=drop
                            )

                    if cards and len(cards) == drop:
                        return (cards, target_count, source)
                    return None

                action_committed = False
                for _ in range(3):
                    if expected_side is None:
                        break

                    actor = None
                    payload = None
                    order = [
                        expected_side,
                        side_cycle[expected_side],
                        side_cycle[side_cycle[expected_side]],
                    ]
                    for side in order:
                        if side == "me":
                            if self_removed:
                                actor = "me"
                                payload = self_removed
                                break
                        else:
                            candidate = opponent_candidate(side)
                            if candidate is not None:
                                actor = side
                                payload = candidate
                                break

                    if actor is None:
                        break
                    if not sync_to_actor(actor, live_counts, raw_hand):
                        break

                    if actor == "me":
                        visual = played_values.get("me") or ""
                        if visual and visual != payload:
                            print(
                                f"我的出牌校正 >>> 桌面识别 {display_cards(visual)}；"
                                f"手牌差分 {display_cards(payload)}，采用手牌差分"
                            )
                        print(f"微信牌局 >>> 我的出牌: {display_cards(payload)}")
                        self.played_card_signal.emit([self.my_position, payload])
                        self.record_hand_inference_action(self.my_position, payload)
                        apply_action_to_douzero(self.my_position, payload)
                        confirmed_my_hand = "" if self_final_out else live_hand
                        self_removed = None
                        expected_side = "right"
                        action_committed = True
                        if self_final_out or not confirmed_my_hand:
                            print("微信牌局 >>> 我的手牌归零，本局结束")
                            self.ai_suggestion_signal.emit([])
                            round_initialized = False
                            expected_side = None
                            break
                    else:
                        cards, new_count, evidence = payload
                        player = player_for_side(actor)
                        if player is None:
                            break
                        label = "左侧" if actor == "left" else "右侧"
                        print(
                            f"微信牌局 >>> {label}出牌: {display_cards(cards)} "
                            f"[剩余 {tracked_remaining[actor]}→{new_count}；{evidence}]"
                        )
                        self.played_card_signal.emit([player, cards])
                        wechat_other_hands_cards_str = remove_chars_from_string(
                            wechat_other_hands_cards_str, cards
                        )
                        self.other_hands_cards_str = wechat_other_hands_cards_str
                        self.card_recorder_signal.emit(wechat_other_hands_cards_str)
                        tracked_remaining[actor] = new_count
                        observed_remaining[actor] = new_count
                        count_desync[actor] = False
                        recent_play[actor] = {"cards": "", "time": 0.0}
                        emit_remaining_counts()
                        self.record_hand_inference_action(player, cards)
                        apply_action_to_douzero(player, cards)
                        expected_side = side_cycle[actor]
                        action_committed = True
                        if new_count == 0:
                            print(f"微信牌局 >>> {label}剩余 0 张，本局结束")
                            self.ai_suggestion_signal.emit([])
                            round_initialized = False
                            expected_side = None
                            break

                # Count OCR also validates state even when no new play is ready.
                if round_initialized:
                    for side, label in (("left", "左侧"), ("right", "右侧")):
                        count = live_counts.get(side)
                        tracked = tracked_remaining.get(side)
                        if count is None or tracked is None:
                            continue
                        if count == tracked:
                            if observed_remaining[side] != count or count_desync[side]:
                                observed_remaining[side] = count
                                count_desync[side] = False
                                print(f"剩余张数 >>> {label}: {count} [画面确认]")
                                emit_remaining_counts()
                        elif count < tracked:
                            candidate = opponent_candidate(side)
                            count_desync[side] = candidate is None
                            if count_desync[side]:
                                observed_remaining[side] = count
                                print(
                                    f"剩余张数校验 >>> {label}画面 {count}，"
                                    f"历史 {tracked}；等待匹配 {tracked - count} 张出牌"
                                )
                                emit_remaining_counts()

                # Explicit Pass remains useful, but it is no longer a single
                # point of failure: later confirmed actions can reconstruct a
                # missed Pass through sync_to_actor().
                if round_initialized:
                    passes = stable_value(
                        "passes", frozenset(recognizer.detect_pass_sides(screenshot)), frames=2
                    )
                    if passes is not None:
                        pass_set = set(passes)
                        for side in pass_latched:
                            if side not in pass_set:
                                pass_latched[side] = False

                        for _ in range(2):
                            if (
                                expected_side in pass_set
                                and not pass_latched[expected_side]
                            ):
                                side = expected_side
                                pass_latched[side] = True
                                record_pass(side, inferred=False)
                            else:
                                break

            except Exception as exc:
                print(f"微信牌局接入异常（不会退出线程）: {exc}")

            await asyncio.sleep(0.35)

        print("微信小程序牌局接入线程已停止")
        print()

    async def before_start(self):
        print('正在检测是否开局...')

        self.in_game_start_screen = await self.gameHelper.check_if_in_game_start_screen()
        while self.worker_runing and not self.in_game_start_screen:
            # if self.auto_play_cards:
            if self.automatic_mode == AutomaticModeEnum.FULL.value:
                success = await self.gameHelper.clickBtn('quick_start_btn')
                if not success:
                    success = await self.gameHelper.clickBtn('continue_game_btn')
            
            print("等待手动进入开始游戏界面...")
            self.in_game_start_screen = await self.gameHelper.check_if_in_game_start_screen()
            time.sleep(1)

        game_started = False
        if self.in_game_start_screen:
            print("您已进入开始游戏界面")

            game_started = await self.gameHelper.check_if_game_started()
            if not game_started:
                print("对局尚未开始")

            while self.worker_runing and not game_started:
                # if self.auto_play_cards:
                if self.automatic_mode == AutomaticModeEnum.FULL.value:
                    await self.gameHelper.clickBtn('start_game_btn')
                
                print("等待手动开始对局...")
                game_started = await self.gameHelper.check_if_game_started()
                time.sleep(1)
        
        if game_started:
            print("对局已开始")
            print()
            self.game_started = True

            await self.autoBidding()
            await self.getThreeCards()
            await self.getMyPosition()
            await self.getMyHandCards()
            await self.autoRedouble()

    async def initial_data(self):
        self.data_initializing = True

        print("正在处理本次牌局数据...")
        self.initOtherPlayerHandCards()
        self.initAllPlayerCardData()

        # 独立于 DouZero 的隐藏手牌概率模型。这里直接复用屏幕识别得到的真实牌局状态。
        self.hand_inference = HandInferenceEngine(
            my_position=self.my_position,
            my_hand_cards=self.my_hand_cards,
            three_landlord_cards=self.three_cards,
        )
        self.refresh_hand_inference()
        print()

        self.play_order = 0 if self.my_position == "landlord" else 1 if self.my_position == "landlord_up" else 2
        playOrderArr = ['我先出牌', '我的下家先出牌', '我的上家先出牌']
        print('出牌顺序: ', playOrderArr[self.play_order])
        print()

        self.create_ai_representer()
        self.env.card_play_init(self.all_player_card_data)

        self.data_initialized = True
        self.data_initializing = False

    async def run_game(self):
        print('准备就绪，玩家开始出牌...')
        print()
        self.card_playing = True
        self.play_order_of_next = self.play_order

        firstOfRound = True
        while self.worker_runing and self.card_playing:
            if self.waiting_for_animation_to_end:
                continue

            if self.play_order_of_next == 0:
                haveAnimation = await self.gameHelper.haveAnimation(AnimationArea.MY_PLAYED_ANIMATION.value)
                if haveAnimation:
                    self.waiting_for_animation_to_end = True
                    print('等待我的动画结束...')
                    print()
                    time.sleep(0.2)

                self.waiting_for_animation_to_end = False
                await self.getMyPlayedCards(firstOfRound)
            elif self.play_order_of_next == 1:
                haveAnimation = await self.gameHelper.haveAnimation(AnimationArea.RIGHT_PLAYED_ANIMATION.value)
                if haveAnimation:
                    self.waiting_for_animation_to_end = True
                    print('等待右侧动画结束...')
                    print()
                    time.sleep(0.2)

                self.waiting_for_animation_to_end = False
                await self.getRightPlayedCards(firstOfRound)
            elif self.play_order_of_next == 2:
                haveAnimation = await self.gameHelper.haveAnimation(AnimationArea.LEFT_PLAYED_ANIMATION.value)
                if haveAnimation:
                    self.waiting_for_animation_to_end = True
                    print('等待左侧动画结束...')
                    print()
                    time.sleep(0.2)
                
                self.waiting_for_animation_to_end = False
                await self.getLeftPlayedCards(firstOfRound)
                
            firstOfRound = False
            if self.env is not None and self.env.game_over:
                time.sleep(3)
                self.round_ended()
                break

            game_overed = await self.gameHelper.check_if_game_overed()
            if game_overed and self.card_playing and self.game_started:
                self.round_ended()
                break

    def round_ended(self):
        self.round_count += 1
        self.reset_status()
        self.reset_ui_status()
        self.reset_ai_env()
        print('本轮对局已结束')
        print()

    def reset_status(self):
        self.in_game_start_screen = False
        self.game_started = False
        self.landlord_confirmed = False
        
        self.three_cards = ''
        self.my_hand_cards = ''

        self.my_position_code = None
        self.my_position = None
        
        self.data_initializing = False
        self.data_initialized = False

        self.play_order = None
        self.play_order_of_next = None

        self.card_playing = False
        
        self.other_hands_cards = []
        self.all_player_card_data = {}

        self.env = None
        self.three_cards_env = None
        self.my_hand_cards_env = None
        self.my_played_cards_env = None
        self.other_played_cards_env = None
        
        self.waiting_for_animation_to_end = False
        self.right_played_completed = False
        self.left_played_completed = False
        self.my_played_completed = False

        self.in_bidding_progress = False
        self.in_redouble_progress = False

        self.my_bidding_status = None
        self.right_bidding_status = None
        self.left_bidding_status = None

        self.ai_suggested_received = False
        self.my_played_card_clicked = False

        self.action_message = None
        self.action_list = None
        self.hand_inference = None

    def reset_ui_status(self):
        self.card_recorder_signal.emit('')
        self.three_cards_signal.emit('')
        self.my_position_signal.emit('')
        self.ai_suggestion_signal.emit([])
        self.bid_win_rate_signal.emit([])
        self.game_win_rate_signal.emit(-1000)
        self.played_card_signal.emit([])
        self.hand_inference_signal.emit({})

    def stop_task(self):
        print("正在停止工作线程...")
        self.worker_runing = False
        self.reset_ui_status()
    
    async def getThreeCards(self):
        print("正在识别三张底牌...")

        while self.worker_runing and len(self.three_cards) != 3:
            self.three_cards = await self.gameHelper.get_three_cards()
            time.sleep(0.2)

        self.three_cards_env = [RealCard2EnvCard[c] for c in list(self.three_cards)]

        print(f"三张底牌：{self.three_cards}")
        print()

        self.three_cards_signal.emit(self.three_cards)

    async def getMyPosition(self):
        print("正在识别我的角色...")

        while self.worker_runing and self.my_position_code is None:
            self.my_position_code = await self.gameHelper.get_my_position()
            time.sleep(0.2)

        self.my_position = PlayerPosition[self.my_position_code]

        print("我的角色：", self.my_position)
        print()

        self.my_position_signal.emit(self.my_position)

    async def getMyHandCards(self):
        print("正在识别我的手牌...")

        success = False
        self.my_hand_cards = await self.gameHelper.get_my_hand_cards()
        if self.my_position_code == 1:
            while self.worker_runing and len(self.my_hand_cards) != 20 and not success:
                self.my_hand_cards = await self.gameHelper.get_my_hand_cards()
                success = len(self.my_hand_cards) == 20
                time.sleep(0.2)

            self.my_hand_cards_env = [RealCard2EnvCard[c] for c in list(self.my_hand_cards)]
        else:
            while self.worker_runing and len(self.my_hand_cards) != 17 and not success:
                self.my_hand_cards = await self.gameHelper.get_my_hand_cards()
                success = len(self.my_hand_cards) == 17
                time.sleep(0.2)

            self.my_hand_cards_env = [RealCard2EnvCard[c] for c in list(self.my_hand_cards)]

        print("我的手牌：", self.my_hand_cards)
        print()

    def initOtherPlayerHandCards(self):
        for i in set(AllEnvCard):
            # 对于每一张牌（i），计算它在整副牌 AllEnvCard 中出现的次数，减去玩家手牌中该牌出现的次数，即为其他玩家手牌中该牌的数量
            self.other_hands_cards.extend([i] * (AllEnvCard.count(i) - self.my_hand_cards_env.count(i)))

        # 将 self.other_player_hand_cards 中的 env牌编码转换为实际的牌面字符，并将它们组合成一个字符串，最后将其反转
        self.other_hands_cards_str = str(''.join([EnvCard2RealCard[c] for c in self.other_hands_cards]))[::-1]
        self.card_recorder_signal.emit(self.other_hands_cards_str)

    def initAllPlayerCardData(self):
        # 这里将牌局的相关数据更新到 self.all_player_card_data 中，包括底牌和每个角色的手牌
        self.all_player_card_data.update({
            'three_landlord_cards':
                self.three_cards_env,
            PlayerPosition[(self.my_position_code + 0) % 3]:
                self.my_hand_cards_env,
            # 上家和下家的手牌是从 self.other_player_hand_cards 中截取的前 17 张和后 17 张牌
            # 具体分配逻辑根据玩家的角色编码 self.my_position_code 和取模操作 (self.my_position_code + n) % 3 决定
            PlayerPosition[(self.my_position_code + 1) % 3]:
                self.other_hands_cards[0:17] if (self.my_position_code + 1) % 3 != 1 else self.other_hands_cards[17:],
            PlayerPosition[(self.my_position_code + 2) % 3]:
                self.other_hands_cards[0:17] if (self.my_position_code + 1) % 3 == 1 else self.other_hands_cards[17:]
        })

    async def getRightPlayedCards(self, firstOfRound):
        if self.right_played_completed:
            return
        
        rightBuchu = None
        if not firstOfRound:
            rightBuchu = await self.gameHelper.get_right_played_text(template='buchu')
            if rightBuchu is not None:
                print("右侧玩家 >>> 不出牌")
                print()

                rightPosition = PlayerPosition[(self.my_position_code + 1) % 3]
                self.played_card_signal.emit([rightPosition, 'Pass'])
        
        tempArr = []
        if rightBuchu is None:
            for i in range(self.try_num):
                result = await self.gameHelper.get_right_played_cards()
                # print(f"第{i + 1}次尝试获取 >>> 右侧玩家的出牌：{result}")
                # print()
                if result is not None and len(result) > 0:
                    tempArr.append(result)
                else:
                    tempArr.append("")
                time.sleep(0.25)
        
        rightPlayedCards = None
        if len(tempArr) == 3:
            # 特殊情况：只有第一次获取到了，后面牌就消失了，再也获取不到了
            if len(tempArr[0]) > 0 and tempArr[1] == "" and tempArr[2] == "":   
                rightPlayedCards = tempArr[0]
            else:
                # 过滤掉空字符串，至少还要有两次获取到的值
                dataList = list(filter(bool, tempArr))
                if len(dataList) > 1:
                    rightPlayedCards = max(dataList, key=len)
        
        rightPlayed = rightPlayedCards is not None and len(rightPlayedCards) > 0
        if rightPlayed:
            print(f"右侧玩家 >>> 已出牌：{rightPlayedCards}")
            print()

            rightPosition = PlayerPosition[(self.my_position_code + 1) % 3]
            self.played_card_signal.emit([rightPosition, rightPlayedCards])
        
        if (rightBuchu is not None) or rightPlayed:
            if rightPlayed:
                self.other_hands_cards_str = remove_chars_from_string(self.other_hands_cards_str, rightPlayedCards)
                self.card_recorder_signal.emit(self.other_hands_cards_str)
            
            tempData = rightPlayedCards if rightPlayed else ""
            self.record_hand_inference_action(rightPosition, tempData)
            self.other_played_cards_env = [RealCard2EnvCard[c] for c in list(tempData)]
            self.other_played_cards_env.sort()
            self.env.step(self.my_position, self.other_played_cards_env)

            self.right_played_completed = True
            self.left_played_completed = False
            self.play_order_of_next = 2

    async def getLeftPlayedCards(self, firstOfRound):
        if self.left_played_completed:
            return

        leftBuchu = None
        if not firstOfRound:
            leftBuchu = await self.gameHelper.get_left_played_text(template='buchu')
            if leftBuchu is not None:
                print("左侧玩家 >>> 不出牌")
                print()

                leftPosition = PlayerPosition[(self.my_position_code + 2) % 3]
                self.played_card_signal.emit([leftPosition, 'Pass'])

        tempArr = []
        if leftBuchu is None:
            for i in range(self.try_num):
                result = await self.gameHelper.get_left_played_cards()
                # print(f"第{i + 1}次尝试获取 >>> 左侧玩家的出牌：{result}")
                # print()
                if result is not None and len(result) > 0:
                    tempArr.append(result)
                else:
                    tempArr.append("")
                time.sleep(0.25)
        
        leftPlayedCards = None
        if len(tempArr) == 3:
            # 特殊情况：只有第一次获取到了，后面牌就消失了，再也获取不到了
            if len(tempArr[0]) > 0 and tempArr[1] == "" and tempArr[2] == "":
                leftPlayedCards = tempArr[0]
            else:
                # 过滤掉空字符串，至少还要有两次获取到的值
                dataList = list(filter(bool, tempArr))
                if len(dataList) > 1:
                    leftPlayedCards = max(dataList, key=len)

        leftPlayed = leftPlayedCards is not None and len(leftPlayedCards) > 0
        if leftPlayed:
            print(f"左侧玩家 >>> 已出牌：{leftPlayedCards}")
            print()

            leftPosition = PlayerPosition[(self.my_position_code + 2) % 3]
            self.played_card_signal.emit([leftPosition, leftPlayedCards])

        if (leftBuchu is not None) or leftPlayed:
            if leftPlayed:
                self.other_hands_cards_str = remove_chars_from_string(self.other_hands_cards_str, leftPlayedCards)
                self.card_recorder_signal.emit(self.other_hands_cards_str)
            
            tempData = leftPlayedCards if leftPlayed else ""
            self.record_hand_inference_action(leftPosition, tempData)
            self.other_played_cards_env = [RealCard2EnvCard[c] for c in list(tempData)]
            self.other_played_cards_env.sort()
            self.env.step(self.my_position, self.other_played_cards_env)

            self.left_played_completed = True
            self.my_played_completed = False
            self.play_order_of_next = 0
    
    async def getMyPlayedCards(self, firstOfRound):
        if self.my_played_completed:
            return
        
        if not self.ai_suggested_received:
            action_message, action_list = self.env.step(self.my_position, update=False)
            action_list = action_list[:3]

            self.action_message = action_message
            self.action_list = action_list

            self.ai_suggestion_signal.emit(self.action_list)
            self.ai_suggested_received = True

        ai_suggested_play_cards = None
        if self.action_message["action"] == "":
            print(f"AI 建议不出牌")
            print()

            # if self.auto_play_cards:
            if self.automatic_mode == AutomaticModeEnum.FULL.value:
                success = await self.gameHelper.clickBtn('not_play_cards_btn')
                if not success:
                    success = await self.gameHelper.clickBtn('can_not_play_cards_btn')
        else:
            ai_suggested_play_cards = self.action_message["action"]
            print(f"AI 建议出牌：{ai_suggested_play_cards}，胜率：{round(self.action_message['win_rate'], 3)}")
            print()

            # if self.auto_play_cards:
            if self.automatic_mode == AutomaticModeEnum.FULL.value or self.automatic_mode == AutomaticModeEnum.SEMI.value:
                if not self.my_played_card_clicked:
                    await self.gameHelper.clickCards(ai_suggested_play_cards)
                    self.my_played_card_clicked = True
                    time.sleep(0.5)

                    if self.automatic_mode == AutomaticModeEnum.FULL.value:
                        await self.gameHelper.clickBtn('play_cards_btn')
        
        myBuchu = None
        if not firstOfRound:
            myBuchu = await self.gameHelper.get_my_played_text(template='buchu')
            if myBuchu is not None:
                print("我 >>> 不出牌")
                print()

                self.played_card_signal.emit([self.my_position, 'Pass'])

        tempArr = []
        if myBuchu is None:
            for i in range(self.try_num):
                result = await self.gameHelper.get_my_played_cards()
                # print(f"第{i + 1}次尝试获取 >>> 我的出牌：{result}")
                # print()
                if result is not None and len(result) > 0:
                    tempArr.append(result)
                else:
                    tempArr.append("")
                time.sleep(0.25)
        
        myPlayedCards = None
        if len(tempArr) == 3:
            # 特殊情况：只有第一次获取到了，后面牌就消失了，再也获取不到了
            if len(tempArr[0]) > 0 and tempArr[1] == "" and tempArr[2] == "":
                myPlayedCards = tempArr[0]
            else:
                # 过滤掉空字符串，至少还要有两次获取到的值
                dataList = list(filter(bool, tempArr))
                if len(dataList) > 1:
                    myPlayedCards = max(dataList, key=len)
        
        myPlayed = myPlayedCards is not None and len(myPlayedCards) > 0
        if myPlayed:
            print(f"我 >>> 已出牌：{myPlayedCards}")
            print()

            self.played_card_signal.emit([self.my_position, myPlayedCards])
        
        if (myBuchu is not None) or myPlayed:
            tempData = myPlayedCards if myPlayed else ""
            self.record_hand_inference_action(self.my_position, tempData)
            self.my_played_cards_env = [RealCard2EnvCard[c] for c in list(tempData)]
            self.my_played_cards_env.sort()
            self.env.step(self.my_position, self.my_played_cards_env)

            my_hand_cards_str = ''.join([EnvCard2RealCard[c] for c in self.env.info_sets[self.my_position].player_hand_cards])
            if len(my_hand_cards_str) == 0:
                self.round_ended()
                return
            
            self.my_played_completed = True
            self.right_played_completed = False
            self.play_order_of_next = 1
            
            self.ai_suggested_received = False
            self.my_played_card_clicked = False

    def refresh_hand_inference(self):
        """重新计算并发送两家隐藏手牌概率；失败时不影响原有牌局线程。"""
        if self.hand_inference is None:
            return

        try:
            result = self.hand_inference.infer()
            self.hand_inference_signal.emit(result)

            summary = HandInferenceEngine.compact_summary(result)
            if summary:
                print("推牌 >>>", summary)
                print()
        except Exception as exc:
            # 推牌器属于辅助层，任何异常都不能打断原项目的识牌和 DouZero 流程。
            print(f"推牌器更新失败: {exc}")
            print()

    def record_hand_inference_action(self, player, cards):
        """记录一次屏幕识别到的公开动作并刷新后验概率。"""
        if self.hand_inference is None:
            return

        try:
            self.hand_inference.observe(player, cards)
            self.refresh_hand_inference()
        except Exception as exc:
            print(f"推牌器记录动作失败: player={player}, cards={cards}, error={exc}")
            print()

    async def autoBidding(self):
        self.in_bidding_progress = True
        self.my_bidding_status = None
        self.right_bidding_status = None
        self.left_bidding_status = None

        win_rate = await self.get_bid_win_rate()
        while self.worker_runing and self.in_bidding_progress:
            await self.get_player_bidding_status()

            # if self.auto_play_cards:
            if self.automatic_mode == AutomaticModeEnum.FULL.value:
                if win_rate > self.bid_threshold:
                    call_success = await self.gameHelper.clickBtn('call_landlord_btn')
                    if call_success:
                        continue
                    scramble_success = await self.gameHelper.clickBtn('scramble_landlord_btn')
                    if scramble_success:
                        continue
                else:
                    not_call_success = await self.gameHelper.clickBtn('not_call_landlord_btn')
                    if not_call_success:
                        continue
                    not_scramble_success = await self.gameHelper.clickBtn('not_scramble_landlord_btn')
                    if not_scramble_success:
                        continue

            three_cards = await self.gameHelper.get_three_cards()
            if len(three_cards) == 3:
                self.in_bidding_progress = False
            
            time.sleep(0.2)
        
        self.check_player_bidding_status()

    async def autoRedouble(self):
        self.in_redouble_progress = True
        win_rate = await self.get_game_win_rate()
        while self.worker_runing and self.in_redouble_progress:
            # if self.auto_play_cards:
            if self.automatic_mode == AutomaticModeEnum.FULL.value:
                if win_rate > self.super_redouble_threshold:
                    success = await self.gameHelper.clickBtn('super_redouble_btn')
                    if not success:
                        success = await self.gameHelper.clickBtn('redouble_btn')
                elif win_rate > self.redouble_threshold:
                    await self.gameHelper.clickBtn('redouble_btn')
                else:
                    await self.gameHelper.clickBtn('not_redouble_btn')
                
                self.landlord_confirmed = True
                self.in_redouble_progress = False
            else:
                self.landlord_confirmed = True
                self.in_redouble_progress = False

            time.sleep(0.2)

    async def get_player_bidding_status(self):
        round_num = self.round_count

        if self.my_bidding_status is None:
            myBj = await self.gameHelper.get_my_played_text(template='bujiao')
            if myBj is not None:
                self.my_bidding_status = 0
                self.update_player_bidding_status(round_num)
            
            myJdz = await self.gameHelper.get_my_played_text(template='jiaodizhu')
            if myJdz is not None:
                self.my_bidding_status = 1
                self.update_player_bidding_status(round_num)

            myBq = await self.gameHelper.get_my_played_text(template='buqiang')
            if myBq is not None:
                self.my_bidding_status = 2
                self.update_player_bidding_status(round_num)

            myQdz = await self.gameHelper.get_my_played_text(template='qiangdizhu')
            if myQdz is not None:
                self.my_bidding_status = 3
                self.update_player_bidding_status(round_num)
        
        if self.right_bidding_status is None:
            rightBj = await self.gameHelper.get_right_played_text(template='bujiao')
            if rightBj is not None:
                self.right_bidding_status = 0
                self.update_player_bidding_status(round_num)
            
            rightJdz = await self.gameHelper.get_right_played_text(template='jiaodizhu')
            if rightJdz is not None:
                self.right_bidding_status = 1
                self.update_player_bidding_status(round_num)
            
            rightBq = await self.gameHelper.get_right_played_text(template='buqiang')
            if rightBq is not None:
                self.right_bidding_status = 2
                self.update_player_bidding_status(round_num)
            
            rightQdz = await self.gameHelper.get_right_played_text(template='qiangdizhu')
            if rightQdz is not None:
                self.right_bidding_status = 3
                self.update_player_bidding_status(round_num)
        
        if self.left_bidding_status is None:
            leftBj = await self.gameHelper.get_left_played_text(template='bujiao')
            if leftBj is not None:
                self.left_bidding_status = 0
                self.update_player_bidding_status(round_num)

            leftJdz = await self.gameHelper.get_left_played_text(template='jiaodizhu')
            if leftJdz is not None:
                self.left_bidding_status = 1
                self.update_player_bidding_status(round_num)

            leftBq = await self.gameHelper.get_left_played_text(template='buqiang')
            if leftBq is not None:
                self.left_bidding_status = 2

            leftQdz = await self.gameHelper.get_left_played_text(template='qiangdizhu')
            if leftQdz is not None:
                self.left_bidding_status = 3

    def check_player_bidding_status(self):
        round_num = self.round_count
        if (round_num in self.player_bidding_status) and all(value == 0 for value in self.player_bidding_status[round_num]):
            if round_num > 1:
                prev_no_bid = all(value == 0 for value in self.player_bidding_status[round_num - 1])
                before_prev_no_bid = all(value == 0 for value in self.player_bidding_status[round_num - 2])
                if prev_no_bid == True and before_prev_no_bid == True:
                    print("连续3局无人叫地主，首家默认为地主")
                    print()
                else:
                    print("本局无人叫地主，重新发牌")
                    print()
                    self.round_ended()
            else:
                print("本局无人叫地主，重新发牌")
                print()
                self.round_ended()

    def update_player_bidding_status(self, round_num):
        self.player_bidding_status[round_num] = [self.my_bidding_status, self.right_bidding_status, self.left_bidding_status]
    
    def create_ai_representer(self):
        AI = [0, 0]
        AI[0] = self.my_position
        AI[1] = DeepAgent(self.my_position, self.model_path_dict[self.my_position])
        self.env = GameEnv(AI)

    def reset_ai_env(self):
        if self.env is not None:
            self.env.game_over = True
            self.env.reset()

    async def get_bid_win_rate(self):
        success = False
        my_hand_cards = await self.gameHelper.get_my_hand_cards()
        while self.worker_runing and len(my_hand_cards) != 17 and not success:
            my_hand_cards = await self.gameHelper.get_my_hand_cards()
            success = len(my_hand_cards) == 17
            time.sleep(0.2)

        bidScore = BidModel.predict_score(my_hand_cards)
        bidWinRate = round(bidScore, 3)
        print(f"预测叫地主胜率：{bidWinRate}")
        print()

        notBidScore = FarmerModel.predict(my_hand_cards, "farmer")
        notBidWinRate = round(notBidScore, 3)
        print(f"预测不叫地主胜率：{notBidWinRate}")
        print()

        self.bid_win_rate_signal.emit([bidWinRate, notBidWinRate])
        return bidWinRate

    async def get_game_win_rate(self):
        if self.my_position_code == 1:
            result = LandlordModel.predict_by_model(self.my_hand_cards, self.three_cards)
            print("本局我是地主，预测胜率：", round(result, 3))
            print()
        elif self.my_position_code == 2:
            result = FarmerModel.predict(self.my_hand_cards, "down")
            print("本局我是农民（地主下家），预测胜率：", round(result, 3))
            print()
        else:
            result = FarmerModel.predict(self.my_hand_cards, "up")
            print("本局我是农民（地主上家），预测胜率：", round(result, 3))
            print()
        
        win_rate = round(result, 3)
        self.game_win_rate_signal.emit(win_rate)
        return win_rate

