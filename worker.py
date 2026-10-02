import asyncio
import hashlib
import json
import os
import time
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone

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
from douzero.env.move_detector import get_move_type, TYPE_4_BOMB, TYPE_5_KING_BOMB
from douzero.evaluation.deep_agent_new import DeepAgent

from constants import RealCard2EnvCard, EnvCard2RealCard, AllEnvCard, AutomaticModeEnum
from utils import remove_chars_from_string
from inference import HandInferenceEngine
from inference.decision_policy import choose_recommendation
from inference.rollout import snapshot_public_env
from inference.rollout_service import RolloutJob, RolloutService
from runtime_audit import LiveAuditWriter

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
        self.last_hand_inference_result = None
        self.last_inference_duration_seconds = 0.0
        self.last_douzero_duration_seconds = 0.0
        self.last_rollout_export_duration_seconds = 0.0

        self.try_num = 3
        self.round_count = 0

        self.model_path_dict = {
            'landlord': "baselines/resnet/resnet_landlord.ckpt",
            'landlord_up': "baselines/resnet/resnet_landlord_up.ckpt",
            'landlord_down': "baselines/resnet/resnet_landlord_down.ckpt"
        }
        self.ai_agent_cache = {}
        self.rollout_service = None
        self.rollout_service_init_failed = False
        self.session_id = uuid.uuid4().hex
        self.rollout_generation = 0
        self.rollout_posterior_revision = 0
        self.rollout_accepting_results = True
        self.stop_requested_event = threading.Event()
        self.current_round_id = None
        self._live_audit_hook = None
        self.audit_writer = LiveAuditWriter(
            os.path.join("screenshots", "inference_audits")
        )

        LandlordModel.init_model("baselines/resnet/resnet_landlord.ckpt")

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.run_task())
        finally:
            self.rollout_accepting_results = False
            if self.rollout_service is not None:
                try:
                    self.rollout_service.stop(
                        self.config.rollout_shutdown_timeout_seconds
                    )
                except Exception as exc:
                    print(f"后台Rollout停止失败: {exc}")
                self.rollout_service = None
            try:
                self.audit_writer.close(0.75)
            except Exception as exc:
                print(f"审计写入器停止失败: {exc}")
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
        last_desync_report = {"left": None, "right": None}
        recent_play = {
            "left": {"cards": "", "time": 0.0},
            "right": {"cards": "", "time": 0.0},
        }
        recent_self_plays = []
        count_missing_frames = {"left": 0, "right": 0}
        self_hand_missing_frames = 0
        pass_latched = {"left": False, "right": False, "me": False}
        last_state = {
            "three_cards": None,
            "position_code": None,
        }
        wechat_other_hands_cards_str = ""
        douzero_initial_data = None
        douzero_players = None
        douzero_history = []
        douzero_paused_reason = None
        suggestion_audit = []
        pending_round_audit = None
        audit_event_seq = 0
        audit_events = []
        audit_status = "idle"
        suggestion_seq = 0
        active_suggestion_id = None
        inference_dirty = False
        last_init_diag_signature = None
        last_init_diag_time = 0.0
        scene_blank_frames = 0
        round_boundary_seen = False
        midgame_wait_announced = False
        last_successful_screenshot_monotonic = 0.0
        previous_successful_screenshot_monotonic = None
        last_frame_gap_seconds = None
        max_frame_gap_seconds = 0.0
        last_audit_error_reported = None
        last_rollout_service_error_reported = None

        side_cycle = {"me": "right", "right": "left", "left": "me"}
        landlord_start_side = {0: "right", 1: "me", 2: "left"}

        def display_cards(cards):
            if not cards:
                return "-"
            display_map = {"D": "大王", "X": "小王", "T": "10"}
            return " ".join(display_map.get(card, card) for card in cards)

        def is_legal_play(cards):
            if not cards:
                return False
            try:
                env_cards = sorted([RealCard2EnvCard[card] for card in cards])
                return get_move_type(env_cards).get("type") != 15
            except Exception:
                return False

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

        def build_live_audit_payload(event_type, extra=None):
            history = []
            if self.hand_inference is not None:
                history = [
                    {
                        "player": player,
                        "action": cards if cards else "Pass",
                    }
                    for player, cards in self.hand_inference.history
                ]
            elif douzero_history:
                history = [
                    {
                        "player": player,
                        "action": cards if cards else "Pass",
                    }
                    for player, cards in douzero_history
                ]

            payload = {
                "format": "wechat_live_inference_audit_v5",
                "schema_version": 5,
                "session_id": self.session_id,
                "round_id": self.current_round_id,
                "status": audit_status,
                "event_type": str(event_type),
                "event_seq": audit_event_seq,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
                "events": list(audit_events),
                "generation_id": self.rollout_generation,
                "posterior_revision": self.rollout_posterior_revision,
                "my_position": self.my_position,
                "initial_my_hand": (
                    ""
                    if self.hand_inference is None
                    else "".join(self.hand_inference.initial_my_hand)
                ),
                "three_landlord_cards": (
                    ""
                    if self.hand_inference is None
                    else "".join(
                        self.hand_inference.three_landlord_cards
                    )
                ),
                "public_history": history,
                "tracked_remaining": dict(tracked_remaining),
                "observed_remaining": dict(observed_remaining),
                "count_desync": dict(count_desync),
                "inference": self.last_hand_inference_result,
                "douzero_history": [
                    {
                        "player": player,
                        "action": cards if cards else "Pass",
                    }
                    for player, cards in douzero_history
                ],
                "suggestion_audit": list(suggestion_audit),
                "posterior_control": {
                    "residual_behavior_strength": (
                        self.config.inference_residual_behavior_strength
                    ),
                    "residual_behavior_temperature": (
                        self.config.inference_residual_behavior_temperature
                    ),
                    "rollout_enabled": bool(self.config.rollout_enabled),
                    "rollout_shadow_mode": bool(
                        self.config.rollout_shadow_mode
                    ),
                    "rollout_device": self.config.rollout_device,
                    "rollout_max_worlds": self.config.rollout_max_worlds,
                    "rollout_min_worlds": self.config.rollout_min_worlds,
                    "rollout_time_budget_seconds": (
                        self.config.rollout_time_budget_seconds
                    ),
                    "hidden_info_audit": "pending_p1",
                },
                "performance": {
                    "last_frame_gap_seconds": last_frame_gap_seconds,
                    "max_frame_gap_seconds": max_frame_gap_seconds,
                    "last_inference_seconds": (
                        self.last_inference_duration_seconds
                    ),
                    "last_douzero_seconds": (
                        self.last_douzero_duration_seconds
                    ),
                    "last_rollout_export_seconds": (
                        self.last_rollout_export_duration_seconds
                    ),
                },
                "rollout_service": (
                    None
                    if self.rollout_service is None
                    else self.rollout_service.status()
                ),
                "audit_writer": self.audit_writer.status(
                    self.current_round_id
                ),
            }
            if extra:
                payload["event_detail"] = dict(extra)
            return payload

        def persist_live_audit(event_type, extra=None):
            nonlocal audit_event_seq, audit_status
            nonlocal last_audit_error_reported
            if not self.current_round_id:
                return False

            writer_status = self.audit_writer.status(
                self.current_round_id
            )
            writer_error = writer_status.get("last_error")
            if (
                writer_error
                and writer_error != last_audit_error_reported
            ):
                print(f"实时审计写入异常 >>> {writer_error}")
                last_audit_error_reported = writer_error
            elif not writer_error:
                last_audit_error_reported = None

            if event_type == "init":
                audit_status = "active"
            elif event_type in ("round_end", "settlement_started"):
                audit_status = "settling"
            elif event_type in (
                "stop_requested",
                "stop",
                "round_reset",
            ):
                audit_status = "interrupted"

            audit_event_seq += 1
            event = {
                "event_seq": audit_event_seq,
                "event_type": str(event_type),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
                "generation_id": self.rollout_generation,
                "posterior_revision": self.rollout_posterior_revision,
                "detail": dict(extra or {}),
            }
            audit_events.append(event)

            payload = build_live_audit_payload(event_type, extra)
            payload["event_seq"] = audit_event_seq
            return self.audit_writer.submit_live(
                self.current_round_id,
                audit_event_seq,
                payload,
            )

        def bump_generation(reason):
            self.rollout_generation += 1
            if self.rollout_service is not None:
                self.rollout_service.invalidate(
                    self.rollout_generation,
                    reason=reason,
                )

        def set_desync(side, value, reason):
            value = bool(value)
            old = bool(count_desync.get(side))
            count_desync[side] = value
            if old == value:
                return False
            bump_generation(
                f"desync_{'enter' if value else 'exit'}:{side}:{reason}"
            )
            persist_live_audit(
                "desync_enter" if value else "desync_exit",
                {
                    "side": side,
                    "reason": reason,
                    "value": value,
                },
            )
            if not value:
                last_desync_report[side] = None
            return True

        def drain_rollout_results():
            nonlocal last_rollout_service_error_reported
            if self.rollout_service is None:
                return

            service_status = self.rollout_service.status()
            service_error = service_status.get("fatal_error")
            if (
                service_error
                and service_error
                != last_rollout_service_error_reported
            ):
                print(
                    "后台Rollout服务已降级关闭 >>> "
                    f"{service_error}"
                )
                last_rollout_service_error_reported = service_error
            elif not service_error:
                last_rollout_service_error_reported = None

            for result in self.rollout_service.drain_results(8):
                record = next(
                    (
                        item
                        for item in reversed(suggestion_audit)
                        if item.get("suggestion_id")
                        == result.suggestion_id
                    ),
                    None,
                )

                expected_hash = (
                    None if record is None
                    else record.get("rollout_state_hash")
                )
                stale_reason = None
                now_monotonic = time.monotonic()
                service_epoch = self.rollout_service.service_epoch
                created_monotonic = (
                    None if record is None
                    else record.get("rollout_created_monotonic")
                )
                expected_dispatch_id = (
                    None if record is None
                    else record.get("rollout_dispatch_id")
                )
                expected_service_epoch = (
                    None if record is None
                    else record.get("rollout_service_epoch")
                )

                if not self.rollout_accepting_results:
                    stale_reason = "not_accepting_results"
                elif result.service_epoch != service_epoch:
                    stale_reason = "service_epoch_mismatch"
                elif expected_service_epoch != result.service_epoch:
                    stale_reason = "suggestion_service_epoch_mismatch"
                elif expected_dispatch_id != result.dispatch_id:
                    stale_reason = "dispatch_id_mismatch"
                elif (
                    created_monotonic is not None
                    and now_monotonic - float(created_monotonic)
                    > self.config.rollout_max_job_age_seconds
                ):
                    stale_reason = "result_ttl_expired"
                elif (
                    last_successful_screenshot_monotonic <= 0
                    or now_monotonic
                    - last_successful_screenshot_monotonic
                    > self.config.rollout_result_max_screen_age_seconds
                ):
                    stale_reason = "screen_stale"
                elif result.session_id != self.session_id:
                    stale_reason = "session_mismatch"
                elif result.round_id != self.current_round_id:
                    stale_reason = "round_mismatch"
                elif result.generation_id != self.rollout_generation:
                    stale_reason = "generation_mismatch"
                elif (
                    result.posterior_revision
                    != self.rollout_posterior_revision
                ):
                    stale_reason = "posterior_revision_mismatch"
                elif record is None:
                    stale_reason = "suggestion_not_found"
                elif result.suggestion_id != active_suggestion_id:
                    stale_reason = "suggestion_not_current"
                elif expected_hash != result.state_hash:
                    stale_reason = "state_hash_mismatch"
                elif any(count_desync.values()):
                    stale_reason = "desync"
                elif (
                    not round_initialized
                    or self.env is None
                    or self.env.game_over
                    or self.env.acting_player_position
                    != self.my_position
                ):
                    stale_reason = "state_no_longer_actionable"

                if stale_reason is not None:
                    if record is not None:
                        record["rollout_status"] = "stale_discarded"
                        record["rollout_stale_reason"] = stale_reason
                    persist_live_audit(
                        "rollout_stale",
                        {
                            "suggestion_id": result.suggestion_id,
                            "job_id": result.job_id,
                            "status": result.status,
                            "reason": stale_reason,
                        },
                    )
                    continue

                try:
                    payload = json.loads(result.payload_json or "{}")
                except Exception:
                    payload = {}

                record["rollout_status"] = result.status
                record["rollout"] = payload
                record["rollout_error"] = result.error

                candidates = payload.get("candidates", [])
                if candidates and result.status in ("ok", "partial"):
                    parts = []
                    for item in candidates:
                        action = item.get("action")
                        shown_action = (
                            "不出"
                            if action == "Pass"
                            else display_cards(action)
                        )
                        parts.append(
                            f"{shown_action}:整局值"
                            f"{item.get('rollout_value', 0):.0%}/"
                            f"终局{item.get('terminal_ratio', 0):.0%}/"
                            f"控权{item.get('control_share', 0):.0%}"
                        )
                    print(
                        "走向评估[后台] >>> "
                        + " | ".join(parts)
                        + f" [状态{result.status}，世界"
                        f"{payload.get('worlds_completed', 0)}，"
                        f"{payload.get('elapsed_seconds', 0):.2f}s]"
                    )

                rollout_event = (
                    "rollout_completed"
                    if result.status in ("ok", "partial")
                    else "rollout_cancelled"
                    if result.status in (
                        "cancelled",
                        "expired",
                        "superseded",
                        "deadline",
                    )
                    else "rollout_insufficient"
                    if result.status == "insufficient_worlds"
                    else "rollout_failed"
                )
                persist_live_audit(
                    rollout_event,
                    {
                        "suggestion_id": result.suggestion_id,
                        "job_id": result.job_id,
                        "dispatch_id": result.dispatch_id,
                        "status": result.status,
                        "error": result.error,
                    },
                )

        def schedule_rollout(
            suggestion_record,
            final_action,
            raw_actions,
            ess_ratio,
        ):
            export_started = time.perf_counter()
            if (
                self.rollout_service is None
                or self.hand_inference is None
                or not self.current_round_id
                or not self.rollout_accepting_results
                or any(count_desync.values())
                or ess_ratio < self.config.rollout_min_ess_ratio
            ):
                return False

            # Long shadow experiments start only after at least one live audit
            # revision is durably visible.
            if not self.audit_writer.is_persisted(
                self.current_round_id,
                1,
            ):
                suggestion_record["rollout_status"] = (
                    "skipped_no_audit_checkpoint"
                )
                return False

            worlds = self.hand_inference.posterior_worlds(
                self.config.rollout_max_worlds,
                allow_infer=False,
            )
            if not worlds:
                suggestion_record["rollout_status"] = (
                    "skipped_no_posterior_cache"
                )
                return False

            candidates = []
            for action in [final_action] + [
                item[0] for item in raw_actions[:3]
            ]:
                action = str(action)
                if action not in candidates:
                    candidates.append(action)
                if len(candidates) >= 3:
                    break

            public_snapshot = snapshot_public_env(
                self.env,
                self.my_position,
            )
            hash_payload = {
                "public": public_snapshot,
                "candidates": candidates,
                "posterior_revision": self.rollout_posterior_revision,
                "model_paths": self.model_path_dict,
                "rollout_config": {
                    "device": self.config.rollout_device,
                    "max_worlds": self.config.rollout_max_worlds,
                    "min_worlds": self.config.rollout_min_worlds,
                    "max_steps": self.config.rollout_max_steps,
                    "time_budget_seconds": (
                        self.config.rollout_time_budget_seconds
                    ),
                },
            }
            state_hash = hashlib.sha256(
                json.dumps(
                    hash_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()

            job_id = uuid.uuid4().hex
            suggestion_record["rollout_state_hash"] = state_hash
            suggestion_record["rollout_job_id"] = job_id
            suggestion_record["rollout_created_monotonic"] = (
                time.monotonic()
            )
            suggestion_record["rollout_service_epoch"] = (
                self.rollout_service.service_epoch
            )
            suggestion_record["rollout_status"] = "submitting"

            job = RolloutJob(
                session_id=self.session_id,
                round_id=self.current_round_id,
                generation_id=self.rollout_generation,
                posterior_revision=self.rollout_posterior_revision,
                job_id=job_id,
                suggestion_id=suggestion_record["suggestion_id"],
                state_hash=state_hash,
                created_at_utc=(
                    datetime.utcnow().isoformat(timespec="milliseconds")
                    + "Z"
                ),
                created_monotonic=time.monotonic(),
                public_snapshot_json=json.dumps(
                    public_snapshot,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                worlds_json=json.dumps(
                    worlds,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                candidates=tuple(candidates),
                my_position=self.my_position,
                posterior_ess_ratio=float(ess_ratio),
                model_paths=tuple(sorted(self.model_path_dict.items())),
                max_worlds=self.config.rollout_max_worlds,
                min_worlds=self.config.rollout_min_worlds,
                max_steps=self.config.rollout_max_steps,
                time_budget_seconds=(
                    self.config.rollout_time_budget_seconds
                ),
                max_job_age_seconds=(
                    self.config.rollout_max_job_age_seconds
                ),
                device=self.config.rollout_device,
                cpu_threads=self.config.rollout_cpu_threads,
            )
            dispatch_id = self.rollout_service.submit(job)
            if dispatch_id is None:
                suggestion_record["rollout_status"] = "submit_failed"
                return False

            suggestion_record["rollout_dispatch_id"] = dispatch_id
            suggestion_record["rollout_status"] = "submitted"
            self.last_rollout_export_duration_seconds = (
                time.perf_counter() - export_started
            )
            suggestion_record["rollout_export_seconds"] = (
                self.last_rollout_export_duration_seconds
            )
            persist_live_audit(
                "rollout_submitted",
                {
                    "suggestion_id": suggestion_record["suggestion_id"],
                    "job_id": job_id,
                    "dispatch_id": dispatch_id,
                    "generation_id": self.rollout_generation,
                    "posterior_revision": (
                        self.rollout_posterior_revision
                    ),
                },
            )
            return True

        self._live_audit_hook = persist_live_audit

        def observe_hand_inference_action(player, cards):
            nonlocal inference_dirty
            if self.hand_inference is None:
                return False
            try:
                self.hand_inference.observe(player, cards)
                inference_dirty = True
                return True
            except Exception as exc:
                print(
                    "推牌器记录动作失败: "
                    f"player={player}, cards={cards}, error={exc}"
                )
                return False

        def flush_frame_inference():
            nonlocal inference_dirty
            if not inference_dirty:
                return
            inference_dirty = False
            self.refresh_hand_inference()

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
            bump_generation(f"public_pass:{player}")
            self.played_card_signal.emit([player, "Pass"])
            observe_hand_inference_action(player, "")
            apply_action_to_douzero(player, "")
            expected_side = side_cycle[side]
            persist_live_audit(
                "pass",
                {
                    "player": player,
                    "side": side,
                    "inferred": bool(inferred),
                },
            )
            return True

        def pause_douzero(reason):
            nonlocal douzero_paused_reason
            if douzero_paused_reason != reason:
                print(f"DouZero建议暂停 >>> {reason}")
                bump_generation(f"douzero_pause:{reason}")
                persist_live_audit(
                    "pause",
                    {"reason": str(reason)},
                )
            douzero_paused_reason = reason
            self.ai_suggestion_signal.emit([("__PAUSED__", "-", "-")])

        def emit_douzero_suggestion_if_my_turn():
            nonlocal douzero_paused_reason, suggestion_seq
            nonlocal active_suggestion_id
            if (
                self.stop_requested_event.is_set()
                or not self.rollout_accepting_results
            ):
                return
            if any(count_desync.values()):
                pause_douzero("等待对手出牌与剩余张数完成匹配")
                return
            if self.env is None or self.env.game_over:
                self.ai_suggestion_signal.emit([])
                return
            if self.env.acting_player_position != self.my_position:
                self.ai_suggestion_signal.emit([])
                return

            try:
                douzero_started = time.perf_counter()
                action_message, action_list = self.env.step(
                    self.my_position, action=None, update=False
                )
                self.last_douzero_duration_seconds = (
                    time.perf_counter() - douzero_started
                )
                self.action_message = action_message

                if not action_list:
                    self.action_list = []
                    self.ai_suggestion_signal.emit([])
                    douzero_paused_reason = None
                    return

                inference_result = self.last_hand_inference_result or {}
                ess_ratio = float(
                    inference_result.get("effective_sample_ratio", 0.0) or 0.0
                )
                raw_actions = list(action_list)
                raw_model_rank = {
                    action: rank
                    for rank, (action, _) in enumerate(raw_actions, start=1)
                }
                raw_top_action, raw_top_score = raw_actions[0]

                # game_new may deliberately replace the raw network top action
                # with a deterministic direct-finish / finish-path action.  The
                # old WeChat display discarded that choice by rebuilding the UI
                # from action_list alone.  Preserve it before belief reranking.
                env_selected_action = ""
                env_selected_win_rate = 0.0
                if isinstance(action_message, dict):
                    env_selected_action = str(
                        action_message.get("action", "") or "Pass"
                    )
                    try:
                        env_selected_win_rate = float(
                            action_message.get("win_rate", 0.0) or 0.0
                        )
                    except (TypeError, ValueError):
                        env_selected_win_rate = 0.0

                env_override = False
                env_selected_rank = raw_model_rank.get(env_selected_action)
                if (
                    env_selected_action
                    and env_selected_action != raw_top_action
                    and env_selected_rank is not None
                    and abs(env_selected_win_rate) >= 1000.0
                ):
                    env_override = True
                    print(
                        "建议链路 >>> game_new确定性路径优先: "
                        f"{display_cards(env_selected_action) if env_selected_action != 'Pass' else '不出'} "
                        f"(原模型第{env_selected_rank})"
                    )

                display_pairs = raw_actions[:3]
                if env_override:
                    selected_pair = raw_actions[env_selected_rank - 1]
                    display_pairs = [selected_pair] + [
                        item
                        for item in raw_actions
                        if item[0] != env_selected_action
                    ][:2]

                left_player = player_for_side("left")
                right_player = player_for_side("right")
                profile_by_action = {}
                enriched_by_action = {}
                audit_candidates = []

                for action_text, score_text in display_pairs:
                    model_rank = raw_model_rank.get(action_text, 999)
                    profile = None
                    if self.hand_inference is not None and action_text != "Pass":
                        try:
                            profile = self.hand_inference.response_profile(
                                action_text,
                                max_samples=self.config.risk_adjustment_response_samples,
                            )
                        except Exception as risk_exc:
                            print(f"敌方响应结构计算失败: {risk_exc}")

                    profile_by_action[action_text] = profile
                    total_text = (
                        "-" if profile is None
                        else f"{profile['can_beat']:.0%}"
                    )
                    player_profiles = (
                        {} if profile is None
                        else profile.get("players", {})
                    )
                    left_profile = player_profiles.get(left_player)
                    right_profile = player_profiles.get(right_player)

                    def split_text(side_profile, key):
                        return (
                            "-"
                            if side_profile is None
                            else f"{side_profile[key]:.0%}"
                        )

                    left_ordinary_text = split_text(
                        left_profile, "ordinary_beat"
                    )
                    left_bomb_text = split_text(left_profile, "bomb_only")
                    right_ordinary_text = split_text(
                        right_profile, "ordinary_beat"
                    )
                    right_bomb_text = split_text(right_profile, "bomb_only")

                    enriched = (
                        action_text,
                        score_text,
                        total_text,
                        left_ordinary_text,
                        left_bomb_text,
                        right_ordinary_text,
                        right_bomb_text,
                        model_rank,
                    )
                    enriched_by_action[action_text] = enriched
                    audit_candidates.append({
                        "action": action_text,
                        "model_rank": model_rank,
                        "model_score": float(score_text),
                        "response_pressure": (
                            None if profile is None
                            else profile.get("pressure")
                        ),
                        "response_profile": profile,
                        "physical_sides": {
                            "left": {
                                "player": left_player,
                                "profile": left_profile,
                            },
                            "right": {
                                "player": right_player,
                                "profile": right_profile,
                            },
                        },
                    })

                # Reranking uses posterior pressure rather than can_beat:
                # ordinary answers count fully, while a bomb-only answer is
                # discounted by the inference engine.  It remains a relative
                # safety signal, never a claimed win probability.
                policy_candidates = []
                for action_text, score_text in raw_actions[:3]:
                    profile = profile_by_action.get(action_text)
                    if profile is None and action_text != "Pass":
                        try:
                            profile = self.hand_inference.response_profile(
                                action_text,
                                max_samples=self.config.risk_adjustment_response_samples,
                            ) if self.hand_inference is not None else None
                        except Exception as risk_exc:
                            print(f"候选压力计算失败: {risk_exc}")
                            profile = None
                        profile_by_action[action_text] = profile
                    pressure = (
                        None if profile is None
                        else profile.get("pressure")
                    )
                    policy_candidates.append(
                        (action_text, float(score_text), pressure)
                    )

                def action_is_bomb(action_text):
                    if action_text in (None, "", "Pass"):
                        return False
                    try:
                        action_env = [
                            RealCard2EnvCard[card]
                            for card in str(action_text)
                        ]
                        move_type = get_move_type(action_env).get("type")
                        return move_type in (
                            TYPE_4_BOMB,
                            TYPE_5_KING_BOMB,
                        )
                    except Exception:
                        return False

                decision = choose_recommendation(
                    policy_candidates,
                    ess_ratio=ess_ratio,
                    enabled=self.config.risk_adjustment_enabled,
                    min_ess_ratio=self.config.risk_adjustment_min_ess_ratio,
                    min_risk_gain=self.config.risk_adjustment_min_risk_gain,
                    max_model_gap_fraction=(
                        self.config.risk_adjustment_max_model_gap_fraction
                    ),
                    max_model_rank=self.config.risk_adjustment_max_model_rank,
                    locked_action=(
                        env_selected_action if env_override else None
                    ),
                    locked_model_rank=env_selected_rank,
                    is_bomb_action=action_is_bomb,
                )

                final_action = (
                    decision.action if decision is not None
                    else raw_top_action
                )

                # Ensure the final action is present in the display set.
                if final_action not in enriched_by_action:
                    for action_text, score_text in raw_actions:
                        if action_text != final_action:
                            continue
                        profile = profile_by_action.get(action_text)
                        total_text = (
                            "-" if profile is None
                            else f"{profile['can_beat']:.0%}"
                        )
                        player_profiles = (
                            {} if profile is None
                            else profile.get("players", {})
                        )
                        left_profile = player_profiles.get(left_player)
                        right_profile = player_profiles.get(right_player)

                        def split_text_late(side_profile, key):
                            return (
                                "-"
                                if side_profile is None
                                else f"{side_profile[key]:.0%}"
                            )

                        enriched_by_action[action_text] = (
                            action_text,
                            score_text,
                            total_text,
                            split_text_late(left_profile, "ordinary_beat"),
                            split_text_late(left_profile, "bomb_only"),
                            split_text_late(right_profile, "ordinary_beat"),
                            split_text_late(right_profile, "bomb_only"),
                            raw_model_rank.get(action_text, 999),
                        )
                        break

                ordered = []
                if final_action in enriched_by_action:
                    ordered.append(enriched_by_action[final_action])
                for action_text, _ in display_pairs:
                    if action_text == final_action:
                        continue
                    item = enriched_by_action.get(action_text)
                    if item is not None:
                        ordered.append(item)
                self.action_list = ordered[:3]
                self.ai_suggestion_signal.emit(self.action_list)
                douzero_paused_reason = None

                final_profile = profile_by_action.get(final_action)
                if self.action_list:
                    (
                        action_text,
                        score_text,
                        total_text,
                        left_ordinary_text,
                        left_bomb_text,
                        right_ordinary_text,
                        right_bomb_text,
                        model_rank,
                    ) = self.action_list[0]
                    shown = (
                        "不出"
                        if action_text == "Pass"
                        else display_cards(action_text)
                    )
                    source_text = {
                        "env_override": "直接出完/路径",
                        "belief_safer": "推牌风险修正",
                        "douzero": "DouZero",
                    }.get(
                        (
                            decision.source
                            if decision is not None
                            else "douzero"
                        ),
                        "DouZero",
                    )
                    print(
                        f"综合建议 >>> {shown} "
                        f"[来源 {source_text}；原模型#{model_rank}] "
                        f"(模型分 {score_text}，敌方总可压 {total_text}；"
                        f"左 普{left_ordinary_text}/炸{left_bomb_text}；"
                        f"右 普{right_ordinary_text}/炸{right_bomb_text})"
                    )
                    if (
                        decision is not None
                        and decision.source == "belief_safer"
                    ):
                        print(
                            "推牌修正 >>> "
                            f"压力下降 {decision.risk_gain:.0%}，"
                            f"模型差距占候选跨度 "
                            f"{decision.model_gap_fraction:.0%}"
                        )

                raw_profile = profile_by_action.get(raw_top_action)
                suggestion_seq += 1
                suggestion_id = (
                    f"{self.current_round_id or 'round'}:"
                    f"{suggestion_seq}"
                )
                active_suggestion_id = suggestion_id
                suggestion_record = {
                    "suggestion_id": suggestion_id,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "generation_id": self.rollout_generation,
                    "posterior_revision": self.rollout_posterior_revision,
                    "history_length": len(douzero_history),
                    "ess_ratio": ess_ratio,
                    "risk_adjustment_enabled": bool(
                        self.config.risk_adjustment_enabled
                    ),
                    "risk_adjustment_applied": bool(
                        decision is not None
                        and decision.source == "belief_safer"
                    ),
                    "decision_source": (
                        decision.source
                        if decision is not None
                        else "douzero"
                    ),
                    "decision_reason": (
                        decision.reason
                        if decision is not None
                        else ""
                    ),
                    "rollout": None,
                    "rollout_status": "not_submitted",
                    "rollout_shadow_mode": bool(
                        self.config.rollout_shadow_mode
                    ),
                    "env_selected": {
                        "action": env_selected_action,
                        "win_rate_marker": env_selected_win_rate,
                        "model_rank": env_selected_rank,
                        "override_applied": env_override,
                    },
                    "raw_top": {
                        "action": raw_top_action,
                        "model_score": float(raw_top_score),
                        "response_risk": (
                            None if raw_profile is None
                            else raw_profile.get("can_beat")
                        ),
                        "response_pressure": (
                            None if raw_profile is None
                            else raw_profile.get("pressure")
                        ),
                        "response_profile": raw_profile,
                    },
                    "adjusted_top": {
                        "action": final_action,
                        "model_rank": raw_model_rank.get(final_action),
                        "response_risk": (
                            None if final_profile is None
                            else final_profile.get("can_beat")
                        ),
                        "response_pressure": (
                            None if final_profile is None
                            else final_profile.get("pressure")
                        ),
                    },
                    "changed_top_action": final_action != raw_top_action,
                    "candidates": sorted(
                        audit_candidates,
                        key=lambda item: item["model_rank"],
                    ),
                }
                suggestion_audit.append(suggestion_record)

                # Persist the user-visible recommendation before submitting the
                # shadow experiment. This preserves causal audit ordering:
                # suggestion -> rollout_submitted -> rollout_result.
                persist_live_audit(
                    "suggestion",
                    {
                        "suggestion_id": suggestion_id,
                        "action": final_action,
                        "rollout_status": suggestion_record.get(
                            "rollout_status"
                        ),
                    },
                )

                # HUD is already updated above. Rollout is queued only after
                # the recommendation is observable, so recognition never waits
                # for posterior simulation.
                schedule_rollout(
                    suggestion_record,
                    final_action,
                    raw_actions,
                    ess_ratio,
                )
            except Exception as exc:
                pause_douzero(f"建议计算失败：{exc}")

        def rebuild_douzero_from_history():
            """Rebuild AI state from confirmed public history.

            Recognition history is the source of truth.  GameEnv is disposable
            derived state, so rebuilding prevents one transient mismatch from
            poisoning all later suggestions.
            """
            nonlocal douzero_paused_reason
            if douzero_players is None or douzero_initial_data is None:
                pause_douzero("初始化数据尚未就绪")
                return False

            try:
                env = GameEnv(douzero_players)
                init_data = {
                    key: list(value)
                    for key, value in douzero_initial_data.items()
                }
                env.card_play_init(init_data)

                for hist_player, hist_cards in douzero_history:
                    if env.game_over:
                        break
                    if env.acting_player_position != hist_player:
                        pause_douzero(
                            f"历史顺序不一致：环境等待 {env.acting_player_position}，"
                            f"历史为 {hist_player}"
                        )
                        return False
                    action_env = sorted(
                        [RealCard2EnvCard[card] for card in hist_cards]
                    )
                    env.step(hist_player, action=action_env, update=True)

                self.env = env
                previous_pause = douzero_paused_reason
                douzero_paused_reason = None
                if previous_pause is not None:
                    bump_generation(
                        f"douzero_resume:{previous_pause}"
                    )
                    persist_live_audit(
                        "resume",
                        {"previous_reason": previous_pause},
                    )
                return True
            except Exception as exc:
                pause_douzero(f"历史重建失败：{exc}")
                return False

        def apply_action_to_douzero(player, cards):
            douzero_history.append((player, cards))
            return rebuild_douzero_from_history()

        def queue_round_audit(reason):
            nonlocal pending_round_audit
            if self.hand_inference is None or not self.current_round_id:
                return

            persist_live_audit(
                "round_end",
                {"reason": str(reason)},
            )
            now = time.monotonic()
            payload = build_live_audit_payload(
                "round_end",
                {"reason": str(reason)},
            )
            payload["event_seq"] = audit_event_seq
            payload["reason"] = reason
            payload["ended_at"] = datetime.now().isoformat(
                timespec="seconds"
            )

            pending_round_audit = {
                # Settlement screenshots are supplemental. The live JSON was
                # already persisted during play and survives if capture fails.
                "capture_times": [now + 0.8, now + 1.6, now + 2.4],
                "screenshots": [],
                "audit_stamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                "round_id": self.current_round_id,
                "event_seq": audit_event_seq,
                "payload": payload,
            }

        def flush_round_audit_if_due(screenshot):
            nonlocal pending_round_audit
            if pending_round_audit is None:
                return

            capture_times = pending_round_audit.get("capture_times", [])
            if not capture_times or time.monotonic() < capture_times[0]:
                return

            try:
                root = os.path.join("screenshots", "inference_audits")
                os.makedirs(root, exist_ok=True)
                stamp = pending_round_audit["audit_stamp"]
                index = len(pending_round_audit["screenshots"]) + 1
                image_name = f"{stamp}_{index}.png"
                image_path = os.path.join(root, image_name)
                screenshot.save(image_path)
                pending_round_audit["screenshots"].append(image_name)
                capture_times.pop(0)

                pending_round_audit["event_seq"] += 1
                pending_round_audit["payload"].setdefault(
                    "events", []
                ).append({
                    "event_seq": pending_round_audit["event_seq"],
                    "event_type": "settlement_screenshot_saved",
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(timespec="milliseconds"),
                    "generation_id": pending_round_audit[
                        "payload"
                    ].get("generation_id"),
                    "posterior_revision": pending_round_audit[
                        "payload"
                    ].get("posterior_revision"),
                    "detail": {
                        "image": image_name,
                        "index": len(
                            pending_round_audit["screenshots"]
                        ),
                    },
                })

                if capture_times:
                    return

                pending_round_audit["event_seq"] += 1
                pending_round_audit["payload"].setdefault(
                    "events", []
                ).append({
                    "event_seq": pending_round_audit["event_seq"],
                    "event_type": "round_finalized",
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(timespec="milliseconds"),
                    "generation_id": pending_round_audit[
                        "payload"
                    ].get("generation_id"),
                    "posterior_revision": pending_round_audit[
                        "payload"
                    ].get("posterior_revision"),
                    "detail": {
                        "screenshots": len(
                            pending_round_audit["screenshots"]
                        )
                    },
                })

                json_name = f"{stamp}.json"
                payload = dict(pending_round_audit["payload"])
                payload["format"] = "wechat_posterior_control_audit_v5"
                payload["schema_version"] = 5
                payload["status"] = "completed"
                payload["updated_at_utc"] = datetime.now(
                    timezone.utc
                ).isoformat(timespec="milliseconds")
                payload["screenshots"] = list(
                    pending_round_audit["screenshots"]
                )
                payload["event_seq"] = pending_round_audit["event_seq"]

                submitted = self.audit_writer.finalize(
                    pending_round_audit["round_id"],
                    pending_round_audit["event_seq"],
                    payload,
                    json_name,
                )
                if submitted:
                    print(
                        "推牌审计final已提交 >>> "
                        f"{os.path.join(root, json_name)} "
                        f"(结算截图 {len(payload['screenshots'])} 张)"
                    )
                else:
                    print(
                        "推牌审计final未重复提交 >>> "
                        f"{pending_round_audit['round_id']}"
                    )
                pending_round_audit = None
            except Exception as exc:
                print(f"推牌审计保存失败: {exc}")
                pending_round_audit = None

        def reset_round_detection_state(reason):
            """Clear stale per-round recognition state at a visual boundary."""
            nonlocal round_signature, expected_side, confirmed_my_hand
            nonlocal pre_landlord_hand, wechat_other_hands_cards_str
            nonlocal douzero_initial_data, douzero_players, douzero_paused_reason
            nonlocal last_init_diag_signature, last_init_diag_time
            nonlocal self_hand_missing_frames, midgame_wait_announced
            nonlocal active_suggestion_id

            print(f"[ROUND/RESET] {reason}，清理上一局缓存，等待下一局")
            if (
                self.current_round_id
                and pending_round_audit is None
            ):
                persist_live_audit(
                    "round_reset",
                    {"reason": str(reason)},
                )
            bump_generation(f"round_reset:{reason}")
            pending.clear()
            round_signature = None
            expected_side = None
            confirmed_my_hand = None
            pre_landlord_hand = None
            wechat_other_hands_cards_str = ""
            preinit_counts["left"] = None
            preinit_counts["right"] = None
            tracked_remaining["left"] = None
            tracked_remaining["right"] = None
            observed_remaining["left"] = None
            observed_remaining["right"] = None
            count_desync["left"] = False
            count_desync["right"] = False
            last_desync_report["left"] = None
            last_desync_report["right"] = None
            count_missing_frames["left"] = 0
            count_missing_frames["right"] = 0
            self_hand_missing_frames = 0
            for side in recent_play:
                recent_play[side] = {"cards": "", "time": 0.0}
            recent_self_plays.clear()
            for side in pass_latched:
                pass_latched[side] = False
            last_state["three_cards"] = None
            last_state["position_code"] = None
            douzero_history.clear()
            suggestion_audit.clear()
            douzero_initial_data = None
            douzero_players = None
            douzero_paused_reason = None
            active_suggestion_id = None
            self.env = None
            self.hand_inference = None
            self.current_round_id = None
            self.my_position_code = None
            self.my_position = ""
            self.action_list = []
            self.ai_suggestion_signal.emit([])
            self.my_position_signal.emit("")
            self.card_recorder_signal.emit("")
            last_init_diag_signature = None
            last_init_diag_time = 0.0
            midgame_wait_announced = False

        def initialize_round(my_hand, three_cards, position_code):
            nonlocal round_initialized, round_signature, expected_side
            nonlocal confirmed_my_hand, wechat_other_hands_cards_str
            nonlocal douzero_initial_data, douzero_players, douzero_paused_reason
            nonlocal audit_event_seq, audit_status, suggestion_seq
            nonlocal active_suggestion_id, inference_dirty

            self.current_round_id = (
                datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                + "_"
                + uuid.uuid4().hex[:6]
            )
            audit_event_seq = 0
            audit_events.clear()
            audit_status = "active"
            suggestion_seq = 0
            active_suggestion_id = None
            inference_dirty = False
            self.rollout_accepting_results = True
            bump_generation("round_init")

            self.my_hand_cards = my_hand
            self.three_cards = three_cards
            self.my_position_code = position_code
            self.my_position = PlayerPosition[position_code]
            confirmed_my_hand = my_hand

            self.hand_inference = HandInferenceEngine(
                my_position=self.my_position,
                my_hand_cards=my_hand,
                three_landlord_cards=three_cards,
                sample_count=self.config.inference_sample_count,
                pass_penalty=self.config.inference_pass_penalty,
                friendly_pass_penalty=self.config.inference_friendly_pass_penalty,
                play_behavior_floor=self.config.inference_play_behavior_floor,
                play_behavior_strength=self.config.inference_play_behavior_strength,
                behavior_temperature=self.config.inference_behavior_temperature,
                min_effective_sample_ratio=self.config.inference_min_effective_sample_ratio,
                residual_behavior_floor=self.config.inference_residual_behavior_floor,
                residual_behavior_strength=self.config.inference_residual_behavior_strength,
                residual_behavior_temperature=self.config.inference_residual_behavior_temperature,
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
            douzero_history.clear()
            suggestion_audit.clear()
            douzero_initial_data = None
            douzero_players = None
            douzero_paused_reason = None
            try:
                self.initOtherPlayerHandCards()
                self.initAllPlayerCardData()
                self.create_ai_representer()
                self._ensure_rollout_service()
                douzero_players = self.env.players
                douzero_initial_data = {
                    key: list(value)
                    for key, value in self.all_player_card_data.items()
                }
                if rebuild_douzero_from_history():
                    print("DouZero只读建议环境已初始化（历史可重建）")
            except Exception as exc:
                self.env = None
                pause_douzero(f"初始化失败：{exc}")

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
            recent_self_plays.clear()

            print()
            print("===== 微信牌局状态已初始化 =====")
            print(f"我的初始手牌({len(my_hand)}): {display_cards(my_hand)}")
            print(f"三张底牌: {display_cards(three_cards)}")
            print(f"我的身份: {self.my_position}")
            print(f"首个行动方: {expected_side}")
            print("==============================")
            print()
            persist_live_audit(
                "init",
                {
                    "initial_hand_count": len(my_hand),
                    "position_code": position_code,
                },
            )
            self.refresh_hand_inference()

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
            if self.stop_requested_event.is_set():
                bump_generation("stop_requested")
                persist_live_audit(
                    "stop_requested",
                    {"reason": "user_stop"},
                )
                self.worker_runing = False
                break

            drain_rollout_results()
            screenshot, _ = await self.screenHelper.getScreenshot()
            if screenshot is None:
                if not missing_reported:
                    print("未找到或无法截图‘腾讯欢乐斗地主’窗口，请保持小程序窗口打开")
                    missing_reported = True
                await asyncio.sleep(0.5)
                continue

            missing_reported = False
            screenshot_now = time.monotonic()
            if previous_successful_screenshot_monotonic is not None:
                last_frame_gap_seconds = (
                    screenshot_now
                    - previous_successful_screenshot_monotonic
                )
                max_frame_gap_seconds = max(
                    max_frame_gap_seconds,
                    last_frame_gap_seconds,
                )
            previous_successful_screenshot_monotonic = screenshot_now
            last_successful_screenshot_monotonic = screenshot_now
            flush_round_audit_if_due(screenshot)
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
                frame_generation_start = self.rollout_generation
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

                visually_blank = (
                    not raw_hand
                    and raw_counts.get("left") is None
                    and raw_counts.get("right") is None
                )
                if visually_blank:
                    scene_blank_frames += 1
                else:
                    scene_blank_frames = 0

                # Fallback end detection: an active round that loses both the
                # hand and both remaining-count badges for ~4 seconds is almost
                # certainly on the settlement/transition screen.  Normal short
                # animations in the supplied recordings lasted only a couple
                # of frames, so require a much longer blank before ending.
                if round_initialized and scene_blank_frames >= 12:
                    print(
                        "[ROUND/END] 连续检测到结算/过渡空白画面，"
                        "按牌局结束处理"
                    )
                    queue_round_audit("visual_settlement")
                    self.ai_suggestion_signal.emit([])
                    round_initialized = False
                    expected_side = None

                # Whether the previous round was fully initialized or the tool
                # was started halfway through a game, a sustained blank scene
                # is the safe boundary at which stale role/bottom/count state
                # can be discarded.  This is what enables continuous multi-
                # round operation without restarting the worker.
                if (
                    not round_initialized
                    and scene_blank_frames >= 6
                    and not round_boundary_seen
                ):
                    reset_round_detection_state("检测到牌局结算/过渡画面")
                    round_boundary_seen = True

                if (
                    round_boundary_seen
                    and init_hand is not None
                    and len(init_hand) in (17, 20)
                ):
                    print(
                        f"[ROUND/START] 检测到新一局稳定牌面："
                        f"我的手牌 {len(init_hand)} 张，"
                        f"左={raw_counts.get('left')}，右={raw_counts.get('right')}"
                    )
                    round_boundary_seen = False

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

                    raw_bottom = recognizer.recognize_bottom_cards(screenshot)
                    top_bottom = stable_value(
                        "three_cards", raw_bottom, frames=2
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

                    hand_pending = pending.get("init_hand", (None, 0))[1]
                    left_count_pending = pending.get(
                        "live_left_count", (None, 0)
                    )[1]
                    right_count_pending = pending.get(
                        "live_right_count", (None, 0)
                    )[1]
                    role_pending = pending.get(
                        "position_code", (None, 0)
                    )[1]
                    bottom_pending = pending.get(
                        "three_cards", (None, 0)
                    )[1]

                    wait_reasons = []
                    if ready_position is None:
                        wait_reasons.append("身份尚未稳定")
                    if expected_count is None:
                        wait_reasons.append("无法确定目标手牌张数")
                    elif len(ready_hand) != expected_count:
                        if not raw_hand:
                            wait_reasons.append(
                                f"未识别到手牌(目标{expected_count}张)"
                            )
                        elif init_hand is None:
                            wait_reasons.append(
                                f"手牌未稳定(raw={len(raw_hand)}张, "
                                f"连续{hand_pending}/5帧, 目标{expected_count}张)"
                            )
                        else:
                            wait_reasons.append(
                                f"稳定手牌张数不符("
                                f"{len(ready_hand)}/{expected_count})"
                            )
                    if len(ready_three) != 3:
                        if not raw_bottom:
                            wait_reasons.append("未识别到三张底牌")
                        elif top_bottom is None:
                            wait_reasons.append(
                                f"底牌未稳定(raw={display_cards(raw_bottom)}, "
                                f"连续{bottom_pending}/2帧)"
                            )
                        else:
                            wait_reasons.append(
                                f"底牌张数异常({len(top_bottom)}张)"
                            )

                    diag_signature = (
                        raw_hand,
                        init_hand,
                        raw_counts.get("left"),
                        live_counts.get("left"),
                        raw_counts.get("right"),
                        live_counts.get("right"),
                        landlord_side,
                        badge_position,
                        inferred_position,
                        position_code,
                        raw_bottom,
                        top_bottom,
                        tuple(wait_reasons),
                    )
                    if (
                        diag_signature != last_init_diag_signature
                        or now - last_init_diag_time >= 2.0
                    ):
                        print(
                            f"[INIT/HAND] raw={len(raw_hand) if raw_hand else 0} "
                            f"[{display_cards(raw_hand)}] | "
                            f"stable={len(init_hand) if init_hand else 0} "
                            f"[{display_cards(init_hand)}] | "
                            f"连续={hand_pending}/5 | "
                            f"目标={expected_count if expected_count is not None else '-'}"
                        )
                        print(
                            f"[INIT/COUNT] 左 raw={raw_counts.get('left')} "
                            f"stable={live_counts.get('left')} "
                            f"连续={left_count_pending}/4 cache={preinit_counts['left']} | "
                            f"右 raw={raw_counts.get('right')} "
                            f"stable={live_counts.get('right')} "
                            f"连续={right_count_pending}/4 cache={preinit_counts['right']}"
                        )
                        print(
                            f"[INIT/ROLE] badge={landlord_side or '-'} "
                            f"badge_stable={badge_position} "
                            f"inferred={inferred_position} "
                            f"stable={position_code} 连续={role_pending}/3 "
                            f"committed={last_state['position_code']}"
                        )
                        bottom_debug = getattr(
                            recognizer, "last_bottom_debug", None
                        )
                        slot_debug = ""
                        if bottom_debug:
                            parts = []
                            for item in bottom_debug:
                                rank = item.get("rank")
                                rank_text = display_cards(rank) if rank else "-"
                                score = item.get("score")
                                score_text = (
                                    "-" if score is None
                                    else f"{float(score):.2f}"
                                )
                                source = item.get("source", "-")
                                parts.append(
                                    f"S{item.get('slot')}={rank_text}"
                                    f"/{score_text}/{source}"
                                )
                            slot_debug = " | slots " + " ".join(parts)
                        print(
                            f"[INIT/BOTTOM] raw={display_cards(raw_bottom)} "
                            f"stable={display_cards(top_bottom)} "
                            f"连续={bottom_pending}/2 "
                            f"committed={display_cards(last_state['three_cards'])}"
                            f"{slot_debug}"
                        )
                        if wait_reasons:
                            print("[INIT/WAIT] " + "；".join(wait_reasons))
                        else:
                            print("[INIT/READY] 初始化条件全部满足")
                        print()
                        last_init_diag_signature = diag_signature
                        last_init_diag_time = now
                    if (
                        expected_count is not None
                        and len(ready_hand) == expected_count
                        and len(ready_three) == 3
                    ):
                        signature = (ready_hand, ready_three, ready_position)
                        if signature != round_signature:
                            initialize_round(ready_hand, ready_three, ready_position)

                if not round_initialized:
                    ready_position = last_state["position_code"]
                    expected_initial = (
                        20 if ready_position == 1
                        else 17 if ready_position is not None
                        else None
                    )
                    if (
                        not midgame_wait_announced
                        and expected_initial is not None
                        and init_hand is not None
                        and 0 < len(init_hand) < expected_initial
                    ):
                        print(
                            f"[ROUND/MIDGAME] 当前稳定手牌 {len(init_hand)} 张，"
                            f"少于新局初始 {expected_initial} 张；"
                            "判定为中途接入，本局不伪造历史，等待结算后自动接下一局"
                        )
                        midgame_wait_announced = True
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

                self_visual = stable_value(
                    "me_play_candidate", played_values.get("me", ""), frames=2
                )
                if self_visual and is_legal_play(self_visual):
                    if (
                        not recent_self_plays
                        or recent_self_plays[-1]["cards"] != self_visual
                    ):
                        recent_self_plays.append(
                            {"cards": self_visual, "time": now}
                        )
                recent_self_plays[:] = [
                    item
                    for item in recent_self_plays
                    if now - item["time"] <= 6.0
                ]

                # Prefer an action from the currently expected seat.  If a later
                # seat has hard evidence, skipped seats are logically Pass.
                self_removed = None
                self_final_out = False
                self_corrected_after = None
                if live_hand and confirmed_my_hand and live_hand != confirmed_my_hand:
                    self_removed = hand_difference(confirmed_my_hand, live_hand)
                    if self_removed == "":
                        confirmed_my_hand = live_hand
                        self_removed = None
                    elif self_removed and not is_legal_play(self_removed):
                        # The after-hand OCR can temporarily miss a raised card,
                        # e.g. a real KK play appears as an impossible KK7 delta.
                        # Do not trust the *current* table image: it may already
                        # contain a later/stale card.  Search only legal table
                        # plays cached since our previous committed action, and
                        # require them to be a multiset subset of the illegal
                        # hand delta.  Prefer the largest candidate, newest on tie.
                        delta_counter = Counter(self_removed)
                        confirmed_counter = Counter(confirmed_my_hand)
                        candidates = []
                        for item in recent_self_plays:
                            candidate = item["cards"]
                            candidate_counter = Counter(candidate)
                            if not candidate or not is_legal_play(candidate):
                                continue
                            if not all(
                                candidate_counter[card] <= delta_counter[card]
                                for card in candidate_counter
                            ):
                                continue
                            if not all(
                                candidate_counter[card] <= confirmed_counter[card]
                                for card in candidate_counter
                            ):
                                continue
                            candidates.append(item)

                        if candidates:
                            best = max(
                                candidates,
                                key=lambda item: (
                                    len(item["cards"]), item["time"]
                                ),
                            )
                            visual_candidate = best["cards"]
                            corrected_after = remove_chars_from_string(
                                confirmed_my_hand, visual_candidate
                            )
                            print(
                                f"我的出牌差分纠错 >>> 原差分 "
                                f"{display_cards(self_removed)} 非法；"
                                f"近期桌面合法候选 "
                                f"{display_cards(visual_candidate)}，"
                                "采用该牌型并反算剩余手牌"
                            )
                            self_removed = visual_candidate
                            self_corrected_after = corrected_after
                        else:
                            print(
                                f"手牌差分候选未采信 >>> {display_cards(self_removed)} "
                                "不是合法牌型，且近期没有可验证的桌面候选"
                            )
                            self_removed = None

                if (
                    not self_removed
                    and confirmed_my_hand
                    and expected_side == "me"
                ):
                    # Final local plays are easier to prove from the table than
                    # from the hand area: after the last card(s) leave, the UI
                    # can immediately enter settlement animation and briefly
                    # hallucinate hand glyphs.  If the table stably shows every
                    # remaining card as one legal move, that is sufficient hard
                    # evidence that the local player went out.
                    raw_final_cards = recognizer.recognize_my_played(
                        screenshot, expected_count=len(confirmed_my_hand)
                    )
                    final_cards = stable_value(
                        "my_final_play", raw_final_cards, frames=2
                    )
                    if (
                        final_cards
                        and Counter(final_cards) == Counter(confirmed_my_hand)
                        and is_legal_play(final_cards)
                    ):
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

                    if cards and len(cards) == drop and is_legal_play(cards):
                        return (cards, target_count, source)
                    if cards and len(cards) == drop and not is_legal_play(cards):
                        print(
                            f"对手出牌候选未采信 >>> "
                            f"{'左侧' if side == 'left' else '右侧'} "
                            f"{display_cards(cards)} 张数正确但牌型不合法"
                        )
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
                        bump_generation(
                            f"public_action:{self.my_position}:{payload}"
                        )
                        self.played_card_signal.emit([self.my_position, payload])
                        observe_hand_inference_action(
                            self.my_position,
                            payload,
                        )
                        apply_action_to_douzero(
                            self.my_position,
                            payload,
                        )
                        persist_live_audit(
                            "action",
                            {
                                "player": self.my_position,
                                "cards": payload,
                                "side": "me",
                            },
                        )
                        confirmed_my_hand = (
                            ""
                            if self_final_out
                            else self_corrected_after
                            if self_corrected_after is not None
                            else live_hand
                        )
                        self_removed = None
                        self_corrected_after = None
                        recent_self_plays.clear()
                        expected_side = "right"
                        action_committed = True
                        if self_final_out or not confirmed_my_hand:
                            print("微信牌局 >>> 我的手牌归零，本局结束")
                            queue_round_audit("my_hand_zero")
                            self.ai_suggestion_signal.emit([])
                            round_initialized = False
                            expected_side = None
                            reset_round_detection_state("我方手牌归零，本局结束")
                            round_boundary_seen = True
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
                        set_desync(
                            actor,
                            False,
                            "matched_confirmed_action",
                        )
                        recent_play[actor] = {"cards": "", "time": 0.0}
                        emit_remaining_counts()
                        bump_generation(
                            f"public_action:{player}:{cards}"
                        )
                        observe_hand_inference_action(player, cards)
                        apply_action_to_douzero(player, cards)
                        persist_live_audit(
                            "action",
                            {
                                "player": player,
                                "cards": cards,
                                "side": actor,
                                "remaining": new_count,
                                "evidence": evidence,
                            },
                        )
                        expected_side = side_cycle[actor]
                        action_committed = True
                        if new_count == 0:
                            print(f"微信牌局 >>> {label}剩余 0 张，本局结束")
                            queue_round_audit(f"{actor}_remaining_zero")
                            self.ai_suggestion_signal.emit([])
                            round_initialized = False
                            expected_side = None
                            reset_round_detection_state(
                                f"{label}剩余0张，本局结束"
                            )
                            round_boundary_seen = True
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
                                set_desync(
                                    side,
                                    False,
                                    "screen_count_matches_history",
                                )
                                print(f"剩余张数 >>> {label}: {count} [画面确认]")
                                emit_remaining_counts()
                        elif count < tracked:
                            candidate = opponent_candidate(side)
                            desync_changed = set_desync(
                                side,
                                candidate is None,
                                "count_drop_without_matched_play",
                            )
                            if count_desync[side]:
                                observed_remaining[side] = count
                                report_key = (count, tracked)
                                if (
                                    desync_changed
                                    or last_desync_report[side]
                                    != report_key
                                ):
                                    print(
                                        f"剩余张数校验 >>> {label}画面 {count}，"
                                        f"历史 {tracked}；等待匹配 {tracked - count} 张出牌"
                                    )
                                    if (
                                        not desync_changed
                                        and last_desync_report[side]
                                        is not None
                                    ):
                                        persist_live_audit(
                                            "desync_updated",
                                            {
                                                "side": side,
                                                "observed": count,
                                                "tracked": tracked,
                                            },
                                        )
                                    last_desync_report[side] = report_key
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

                # Commit the frame as one coherent public state. Several
                # inferred Pass/actions in one screenshot produce one posterior
                # refresh and one suggestion/rollout job for the final state.
                flush_frame_inference()
                if self.rollout_generation != frame_generation_start:
                    emit_douzero_suggestion_if_my_turn()

            except Exception as exc:
                print(f"微信牌局接入异常（不会退出线程）: {exc}")

            drain_rollout_results()
            await asyncio.sleep(0.35)

        if self.current_round_id:
            persist_live_audit(
                "stop",
                {"interrupted": True},
            )
            self.audit_writer.flush(0.5)
        self._live_audit_hook = None
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
            sample_count=self.config.inference_sample_count,
            pass_penalty=self.config.inference_pass_penalty,
            friendly_pass_penalty=self.config.inference_friendly_pass_penalty,
            play_behavior_floor=self.config.inference_play_behavior_floor,
            play_behavior_strength=self.config.inference_play_behavior_strength,
            behavior_temperature=self.config.inference_behavior_temperature,
            min_effective_sample_ratio=self.config.inference_min_effective_sample_ratio,
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
        self.rollout_accepting_results = False
        self.stop_requested_event.set()

        # The WeChat live thread owns generation/history and performs the
        # actual invalidation at its next safe frame boundary.
        if getattr(self.config, "platform", "") != "wechat_miniapp":
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
            inference_started = time.perf_counter()
            result = self.hand_inference.infer()
            self.last_inference_duration_seconds = (
                time.perf_counter() - inference_started
            )
            self.last_hand_inference_result = result
            self.rollout_posterior_revision += 1
            if self.stop_requested_event.is_set():
                return
            self.hand_inference_signal.emit(result)

            hook = self._live_audit_hook
            if hook is not None:
                try:
                    hook("inference_completed")
                except Exception as audit_exc:
                    print(f"实时审计提交失败: {audit_exc}")

            summary = HandInferenceEngine.compact_summary(result)
            if summary:
                print("推牌 >>>", summary)
                print()
        except Exception as exc:
            if "inference_started" in locals():
                self.last_inference_duration_seconds = (
                    time.perf_counter() - inference_started
                )
            # 推牌器属于辅助层，任何异常都不能打断原项目的识牌和 DouZero 流程。
            print(f"推牌器更新失败: {exc}")
            hook = self._live_audit_hook
            if hook is not None:
                try:
                    hook(
                        "inference_failed",
                        {"error": repr(exc)},
                    )
                except Exception:
                    pass
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
    
    def _get_cached_agent(self, position):
        agent = self.ai_agent_cache.get(position)
        if agent is None:
            agent = DeepAgent(position, self.model_path_dict[position])
            self.ai_agent_cache[position] = agent
        return agent

    def _ensure_rollout_service(self):
        if not getattr(self.config, "rollout_enabled", False):
            return None
        if self.rollout_service is not None:
            return self.rollout_service
        if self.rollout_service_init_failed:
            return None

        try:
            self.rollout_service = RolloutService(
                model_paths=self.model_path_dict,
                device=self.config.rollout_device,
                cpu_threads=self.config.rollout_cpu_threads,
                shutdown_timeout=(
                    self.config.rollout_shutdown_timeout_seconds
                ),
            )
            print(
                "后验Rollout后台服务启动中 >>> "
                f"device={self.config.rollout_device} "
                f"worlds<={self.config.rollout_max_worlds} "
                f"budget={self.config.rollout_time_budget_seconds:.1f}s "
                f"shadow={self.config.rollout_shadow_mode}"
            )
            return self.rollout_service
        except Exception as exc:
            self.rollout_service_init_failed = True
            self.rollout_service = None
            print(
                "后验Rollout后台服务初始化失败，自动降级为DouZero: "
                f"{exc}"
            )
            return None

    def create_ai_representer(self):
        AI = [0, 0]
        AI[0] = self.my_position
        AI[1] = self._get_cached_agent(self.my_position)
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

