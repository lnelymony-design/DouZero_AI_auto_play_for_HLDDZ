from PyQt5 import QtWidgets, QtGui, QtCore

from constants import AutomaticModeEnum, RealCards
from helpers.ScreenHelper import ScreenHelper
from worker import WorkerThread


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.screenHelper = ScreenHelper()
        self.max_width = 778
        self.max_height = 650

        if getattr(self.screenHelper.config, 'platform', '') == 'wechat_miniapp':
            self.setWindowTitle("微信小程序 - 欢乐斗地主 AI 辅助")
        else:
            self.setWindowTitle("QQ 游戏大厅 - 欢乐斗地主 AI 辅助")
        self.setGeometry(320, 160, self.max_width, self.max_height)
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout  = QtWidgets.QVBoxLayout(central_widget)

        self.create_comboBox()
        self.create_card_counter_table()
        self.create_label()
        self.create_other_tables()
        self.create_inference_table()
        self.create_actions()

        self.workerThread = None
        self.observedRemainingCounts = {}
        self.currentMyPosition = None
        self.bid_threshold = 0.6
        self.redouble_threshold = 0.65
        self.super_redouble_threshold = 0.7
        self.mingpai_threshold = 0.95
        if getattr(self.screenHelper.config, 'platform', '') == 'wechat_miniapp':
            self.automatic_mode = AutomaticModeEnum.MANUAL.value
        else:
            self.automatic_mode = AutomaticModeEnum.FULL.value
        self.screenHelper.setWindowSize()

    # 禁用窗口拖拽和缩放
    def resizeEvent(self, event):
        self.setFixedSize(self.max_width, self.max_height)

    def moveEvent(self, event):
        self.move(self.x(), self.y())

    def handle_cbBid_selection_changed(self):
        current_value = self.cbBid.currentData()
        self.bid_threshold = float(current_value)

    def handle_cbRedoubleX2_selection_changed(self):
        current_value = self.cbRedoubleX2.currentData()
        self.redouble_threshold = float(current_value)

    def handle_cbRedoubleX4_selection_changed(self):
        current_value = self.cbRedoubleX4.currentData()
        self.super_redouble_threshold = float(current_value)

    def handle_cbMingpai_selection_changed(self):
        current_value = self.cbMingpai.currentData()
        self.mingpai_threshold = float(current_value)

    def handle_cbMode_selection_changed(self):
        current_value = self.cbMode.currentData()
        self.automatic_mode = AutomaticModeEnum(current_value).value

    def set_status(self, running):
        self.cbBid.setEnabled(not running)
        self.cbRedoubleX2.setEnabled(not running)
        self.cbRedoubleX4.setEnabled(not running)
        self.cbMingpai.setEnabled(not running)
        self.cbMode.setEnabled(not running)

    def handle_startBtn_clicked(self):
        if self.workerThread is None:
            self.workerThread = WorkerThread(self.automatic_mode, self.bid_threshold, self.redouble_threshold, self.super_redouble_threshold, self.mingpai_threshold)
            self.workerThread.card_recorder_signal.connect(self.handle_card_recorder_update)
            self.workerThread.three_cards_signal.connect(self.handle_three_cards_update)
            self.workerThread.my_position_signal.connect(self.handle_my_position_update)
            self.workerThread.ai_suggestion_signal.connect(self.handle_ai_suggestion_update)
            self.workerThread.bid_win_rate_signal.connect(self.handle_bid_win_rate_update)
            self.workerThread.game_win_rate_signal.connect(self.handle_game_win_rate_update)
            self.workerThread.played_card_signal.connect(self.handle_played_card_update)
            self.workerThread.hand_inference_signal.connect(self.handle_hand_inference_update)
            self.workerThread.remaining_count_signal.connect(self.handle_remaining_count_update)

        if not self.workerThread.isRunning():
            self.workerThread.start()
            self.startBtn.setText("停止")
            self.set_status(True)
        else:
            self.stop_worker_thread()

    def stop_worker_thread(self):
        if self.workerThread is None:
            return

        self.workerThread.stop_task()
        if self.workerThread.isRunning():
            self.workerThread.wait(3000)

        # The worker normally stops cooperatively. Keep a bounded fallback so a
        # recognition loop can never make the GUI impossible to close again.
        if self.workerThread.isRunning():
            print("工作线程未在 3 秒内停止，正在强制结束...")
            self.workerThread.terminate()
            self.workerThread.wait(1000)

        self.workerThread = None
        self.startBtn.setText("启动")
        self.set_status(False)

    def closeEvent(self, event):
        self.stop_worker_thread()
        event.accept()

    @staticmethod
    def display_cards(cards):
        if not cards:
            return cards
        display_map = {"D": "大王", "X": "小王", "T": "10"}
        return " ".join(display_map.get(card, card) for card in cards)

    def handle_card_recorder_update(self, result):
        font = QtGui.QFont("微软雅黑", 10, QtGui.QFont.Bold)

        for i in range(15):
            char = RealCards[i]
            num = result.count(char)

            newItem = QtWidgets.QTableWidgetItem(str(num))
            newItem.setFont(font)
            newItem.setTextAlignment(QtCore.Qt.AlignCenter)
            newItem.setBackground(QtGui.QColor('#404040'))
            newItem.setForeground(QtGui.QColor('#ffffff'))

            if num == 4:
                newItem.setForeground(QtGui.QColor("#ff0000"))
            if num == 0:
                newItem.setForeground(QtGui.QColor("#7c7c7c"))
            
            self.cardCounterTable.setItem(1, i, newItem)

    def handle_three_cards_update(self, result):
        if len(result) == 0:
            self.threeCardsLabel.setText('---')
            return
        
        self.threeCardsLabel.setText(self.display_cards(result))

    def handle_my_position_update(self, result):
        if len(result) == 0:
            self.currentMyPosition = None
            self.myPositionLabel.setText('---')
            self._refresh_inference_side_labels()
            return

        self.currentMyPosition = result
        
        posotionTextMap = {
            'landlord_up': '农民（地主上家）',
            'landlord': '地主',
            'landlord_down': '农民（地主下家）',
        }

        self.myPositionLabel.setText(posotionTextMap[result])
        self._refresh_inference_side_labels()

    def handle_ai_suggestion_update(self, result):
        font_content = QtGui.QFont("微软雅黑", 8, QtGui.QFont.Bold)

        if (
            isinstance(result, list)
            and result
            and isinstance(result[0], (list, tuple))
            and result[0]
            and result[0][0] == "__PAUSED__"
        ):
            for row in range(1, 4):
                for col in range(3):
                    value = "状态暂停" if row == 1 and col == 0 else "-"
                    item = QtWidgets.QTableWidgetItem(value)
                    item.setFont(font_content)
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    if row == 1 and col == 0:
                        item.setForeground(QtGui.QColor("#d00000"))
                        item.setToolTip("识别历史与DouZero环境暂时无法一致重建；识牌和概率推断仍在继续")
                    self.suggestionTable.setItem(row, col, item)
            return

        if not result or not isinstance(result, list):
            for row in range(1, 4):
                for col in range(3):
                    item = QtWidgets.QTableWidgetItem("-")
                    item.setFont(font_content)
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.suggestionTable.setItem(row, col, item)
            return

        for i in range(3):
            data = result[i] if len(result) > i else ("-", "-", "-")
            action_text = data[0] if len(data) > 0 else "-"
            score_text = data[1] if len(data) > 1 else "-"
            risk_text = data[2] if len(data) > 2 else "-"

            if isinstance(action_text, str) and action_text not in ("-", "Pass"):
                action_text = self.display_cards(action_text)

            action_item = QtWidgets.QTableWidgetItem(action_text)
            action_item.setFont(font_content)
            action_item.setTextAlignment(QtCore.Qt.AlignCenter)
            action_item.setForeground(QtGui.QColor("#0066cc"))
            self.suggestionTable.setItem(i + 1, 0, action_item)

            score_item = QtWidgets.QTableWidgetItem(str(score_text))
            score_item.setFont(font_content)
            score_item.setTextAlignment(QtCore.Qt.AlignCenter)
            if score_text != "-":
                try:
                    score = float(score_text)
                    if score >= 1:
                        score_item.setForeground(QtGui.QColor("#d00000"))
                    elif score < 0:
                        score_item.setForeground(QtGui.QColor("#808080"))
                except (TypeError, ValueError):
                    pass
            self.suggestionTable.setItem(i + 1, 1, score_item)

            risk_item = QtWidgets.QTableWidgetItem(str(risk_text))
            risk_item.setFont(font_content)
            risk_item.setTextAlignment(QtCore.Qt.AlignCenter)
            risk_item.setToolTip("当前隐藏手牌样本中，敌方至少有一手合法牌可压住该动作的比例；不是胜率，也不代表敌方一定会出")
            if isinstance(risk_text, str) and risk_text.endswith("%"):
                try:
                    risk = float(risk_text[:-1]) / 100.0
                    if risk >= 0.75:
                        risk_item.setForeground(QtGui.QColor("#d00000"))
                    elif risk >= 0.50:
                        risk_item.setForeground(QtGui.QColor("#c26b00"))
                    elif risk <= 0.20:
                        risk_item.setForeground(QtGui.QColor("#008000"))
                except ValueError:
                    pass
            self.suggestionTable.setItem(i + 1, 2, risk_item)

    def handle_bid_win_rate_update(self, result):
        font_content = QtGui.QFont("微软雅黑", 8, QtGui.QFont.Bold)

        if len(result) == 0 or not isinstance(result, list):
            emptyItem1 = QtWidgets.QTableWidgetItem("-")
            emptyItem1.setFont(font_content)
            emptyItem1.setTextAlignment(QtCore.Qt.AlignCenter)
            emptyItem2 = QtWidgets.QTableWidgetItem("-")
            emptyItem2.setFont(font_content)
            emptyItem2.setTextAlignment(QtCore.Qt.AlignCenter)
            self.predictInfoTable.setItem(1, 1, emptyItem1)
            self.predictInfoTable.setItem(2, 1, emptyItem2)
            return
        
        font_content = QtGui.QFont("微软雅黑", 8, QtGui.QFont.Bold)

        bidItem = QtWidgets.QTableWidgetItem(f"{result[0]}")
        bidItem.setFont(font_content)
        bidItem.setTextAlignment(QtCore.Qt.AlignCenter)

        if result[0] >= self.bid_threshold:
            bidItem.setForeground(QtGui.QColor("#ff0000"))
        else:
            bidItem.setForeground(QtGui.QColor("#00FF00"))

        notBidItem = QtWidgets.QTableWidgetItem(f"{result[1]}")
        notBidItem.setFont(font_content)
        notBidItem.setTextAlignment(QtCore.Qt.AlignCenter)
        
        if result[1] >= self.bid_threshold:
            notBidItem.setForeground(QtGui.QColor("#ff0000"))
        else:
            notBidItem.setForeground(QtGui.QColor("#00FF00"))

        self.predictInfoTable.setItem(1, 1, bidItem)
        self.predictInfoTable.setItem(2, 1, notBidItem)
    
    def handle_game_win_rate_update(self, result):
        font_content = QtGui.QFont("微软雅黑", 8, QtGui.QFont.Bold)

        if not result or result == -1000:
            emptyItem = QtWidgets.QTableWidgetItem("-")
            emptyItem.setFont(font_content)
            emptyItem.setTextAlignment(QtCore.Qt.AlignCenter)
            self.predictInfoTable.setItem(3, 1, emptyItem)
            return

        gameWinItem = QtWidgets.QTableWidgetItem(f"{result}")
        gameWinItem.setFont(font_content)
        gameWinItem.setTextAlignment(QtCore.Qt.AlignCenter)
        
        if result >= self.redouble_threshold:
            gameWinItem.setForeground(QtGui.QColor("#ff0000"))
        elif result < self.redouble_threshold:
            gameWinItem.setForeground(QtGui.QColor("#000000"))
        elif result < 0.5:
            gameWinItem.setForeground(QtGui.QColor("#00FF00"))

        self.predictInfoTable.setItem(3, 1, gameWinItem)

    def handle_played_card_update(self, result):
        font_content = QtGui.QFont("微软雅黑", 8, QtGui.QFont.Bold)

        if len(result) == 0 or not isinstance(result, list):
            emptyItem1 = QtWidgets.QTableWidgetItem("-")
            emptyItem1.setFont(font_content)
            emptyItem1.setTextAlignment(QtCore.Qt.AlignCenter)
            emptyItem2 = QtWidgets.QTableWidgetItem("-")
            emptyItem2.setFont(font_content)
            emptyItem2.setTextAlignment(QtCore.Qt.AlignCenter)
            emptyItem3 = QtWidgets.QTableWidgetItem("-")
            emptyItem3.setFont(font_content)
            emptyItem3.setTextAlignment(QtCore.Qt.AlignCenter)
            self.playedCardsTable.setItem(1, 1, emptyItem1)
            self.playedCardsTable.setItem(2, 1, emptyItem2)
            self.playedCardsTable.setItem(3, 1, emptyItem3)
            return

        played_text = result[1] if result[1] == "Pass" else self.display_cards(result[1])
        playedCardRightItem = QtWidgets.QTableWidgetItem(played_text)
        playedCardRightItem.setFont(font_content)
        playedCardRightItem.setTextAlignment(QtCore.Qt.AlignCenter)
        playedCardRightItem.setForeground(QtGui.QColor("#0000FF"))

        if result[0] == 'landlord_up':
            self.playedCardsTable.setItem(1, 1, playedCardRightItem)
        elif result[0] == 'landlord':
            self.playedCardsTable.setItem(2, 1, playedCardRightItem)
        elif result[0] == 'landlord_down':
            self.playedCardsTable.setItem(3, 1, playedCardRightItem)

    def create_inference_table(self):
        title_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("对手手牌推断")
        title.setFont(QtGui.QFont("微软雅黑", 9, QtGui.QFont.Bold))
        title_layout.addWidget(title)
        title_layout.addStretch()

        self.inferenceMetaLabel = QtWidgets.QLabel("样本 - · Pass - · 状态等待")
        self.inferenceMetaLabel.setFont(QtGui.QFont("微软雅黑", 8))
        title_layout.addWidget(self.inferenceMetaLabel)
        self.main_layout.addLayout(title_layout)

        self.inferenceTable = QtWidgets.QTableWidget(self)
        self.inferenceTable.setRowCount(3)
        self.inferenceTable.setColumnCount(9)
        self.inferenceTable.verticalHeader().setVisible(False)
        self.inferenceTable.horizontalHeader().setVisible(False)
        self.inferenceTable.setShowGrid(False)
        self.inferenceTable.setFixedHeight(98)
        self.inferenceTable.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.inferenceTable.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.inferenceTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.inferenceTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        headers = ["对手", "剩余", "大王", "小王", "2", "A", "K", "炸弹", "王炸"]
        widths = [145, 58, 67, 67, 55, 55, 55, 76, 76]
        for col, (header, width) in enumerate(zip(headers, widths)):
            self.inferenceTable.setColumnWidth(col, width)
            item = QtWidgets.QTableWidgetItem(header)
            item.setFont(QtGui.QFont("微软雅黑", 8, QtGui.QFont.Bold))
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.inferenceTable.setItem(0, col, item)

        for row, side in ((1, "左侧"), (2, "右侧")):
            self.inferenceTable.setRowHeight(row, 30)
            name_item = QtWidgets.QTableWidgetItem(side)
            name_item.setFont(QtGui.QFont("微软雅黑", 8, QtGui.QFont.Bold))
            name_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.inferenceTable.setItem(row, 0, name_item)
            for col in range(1, 9):
                item = QtWidgets.QTableWidgetItem("-")
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.inferenceTable.setItem(row, col, item)

        self.main_layout.addWidget(self.inferenceTable)

    def _side_position_map(self):
        mapping = {
            "landlord": {"left": "landlord_up", "right": "landlord_down"},
            "landlord_up": {"left": "landlord_down", "right": "landlord"},
            "landlord_down": {"left": "landlord", "right": "landlord_up"},
        }
        return mapping.get(self.currentMyPosition, {})

    @staticmethod
    def _position_display_name(position):
        return {
            "landlord": "地主",
            "landlord_up": "农民",
            "landlord_down": "农民",
        }.get(position, "-")

    def _refresh_inference_side_labels(self):
        if not hasattr(self, "inferenceTable"):
            return
        side_map = self._side_position_map()
        for row, side_key, side_text in (
            (1, "left", "左侧"),
            (2, "right", "右侧"),
        ):
            position = side_map.get(side_key)
            role = self._position_display_name(position)
            text = side_text if role == "-" else f"{side_text}（{role}）"
            item = QtWidgets.QTableWidgetItem(text)
            item.setFont(QtGui.QFont("微软雅黑", 8, QtGui.QFont.Bold))
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.inferenceTable.setItem(row, 0, item)

    def handle_remaining_count_update(self, result):
        if not isinstance(result, dict):
            return
        self.observedRemainingCounts.update(result)
        self._render_remaining_counts()

    def _render_remaining_counts(self):
        side_map = self._side_position_map()
        for row, side_key in ((1, "left"), (2, "right")):
            position = side_map.get(side_key)
            info = self.observedRemainingCounts.get(position, {}) if position else {}
            count = info.get("count") if isinstance(info, dict) else None
            if count is None:
                continue

            value = f"{count}实" if info.get("observed") else str(count)
            if info.get("desync"):
                value = f"{count}!"

            item = QtWidgets.QTableWidgetItem(value)
            item.setFont(QtGui.QFont("微软雅黑", 8, QtGui.QFont.Bold))
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            if info.get("desync"):
                item.setForeground(QtGui.QColor("#ff0000"))
                item.setToolTip("画面剩余张数与动作历史尚未完成匹配")
            elif info.get("observed"):
                item.setForeground(QtGui.QColor("#0066cc"))
                item.setToolTip("微信画面确认的剩余张数")
            self.inferenceTable.setItem(row, 1, item)

    def handle_hand_inference_update(self, result):
        if not result or not result.get("players"):
            self.inferenceMetaLabel.setText("样本 - · Pass - · 状态等待")
            for row in (1, 2):
                for col in range(1, 9):
                    item = QtWidgets.QTableWidgetItem("-")
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.inferenceTable.setItem(row, col, item)
            return

        desync = any(
            isinstance(info, dict) and info.get("desync")
            for info in self.observedRemainingCounts.values()
        )
        status = "状态需校验" if desync else "状态正常"
        effective = result.get("effective_samples", "-")
        self.inferenceMetaLabel.setText(
            f"样本 {result.get('samples', '-')} / 有效 {effective} · "
            f"Pass {result.get('pass_evidence_count', 0)} · {status}"
        )

        self._refresh_inference_side_labels()
        side_map = self._side_position_map()
        players = result.get("players", {})

        for row, side_key in ((1, "left"), (2, "right")):
            position = side_map.get(side_key)
            data = players.get(position) if position else None
            if not data:
                for col in range(1, 9):
                    item = QtWidgets.QTableWidgetItem("-")
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.inferenceTable.setItem(row, col, item)
                continue

            observed_info = self.observedRemainingCounts.get(position, {})
            observed_count = observed_info.get("count")
            if observed_count is not None:
                remaining_text = (
                    f"{observed_count}!"
                    if observed_info.get("desync")
                    else f"{observed_count}实"
                )
            else:
                remaining_text = str(data.get("remaining_count", "-"))

            values = [
                remaining_text,
                self._format_inference_probability(data, "D"),
                self._format_inference_probability(data, "X"),
                self._format_inference_probability(data, "2"),
                self._format_inference_probability(data, "A"),
                self._format_inference_probability(data, "K"),
                f"{data.get('any_bomb', 0):.0%}",
                f"{data.get('rocket', 0):.0%}",
            ]

            for col, value in enumerate(values, start=1):
                item = QtWidgets.QTableWidgetItem(value)
                item.setFont(QtGui.QFont("微软雅黑", 8, QtGui.QFont.Bold))
                item.setTextAlignment(QtCore.Qt.AlignCenter)

                if col == 1 and value.endswith("!"):
                    item.setForeground(QtGui.QColor("#ff0000"))
                elif col == 1 and value.endswith("实"):
                    item.setForeground(QtGui.QColor("#0066cc"))
                elif col >= 2 and value.endswith("%"):
                    try:
                        p = float(value[:-1]) / 100.0
                        if p >= 0.75:
                            item.setForeground(QtGui.QColor("#d00000"))
                        elif p >= 0.50:
                            item.setForeground(QtGui.QColor("#c26b00"))
                        elif p <= 0.10:
                            item.setForeground(QtGui.QColor("#808080"))
                    except ValueError:
                        pass

                    card_by_col = {2: "D", 3: "X", 4: "2", 5: "A", 6: "K"}
                    card = card_by_col.get(col)
                    if card:
                        stats = data.get("cards", {}).get(card, {})
                        card_name = {"D": "大王", "X": "小王"}.get(card, card)
                        item.setToolTip(
                            f"{card_name}：至少1张 {stats.get('one_plus', 0):.0%}\n"
                            f"至少对子 {stats.get('pair_plus', 0):.0%}\n"
                            f"至少三张 {stats.get('triple_plus', 0):.0%}\n"
                            f"四张炸弹 {stats.get('bomb', 0):.0%}"
                        )
                self.inferenceTable.setItem(row, col, item)

        self._render_remaining_counts()

    @staticmethod
    def _format_inference_probability(player_data, card):
        probability = (
            player_data.get("cards", {})
            .get(card, {})
            .get("one_plus", 0)
        )
        return f"{probability:.0%}"

    def create_comboBox(self):
        combo_layout = QtWidgets.QHBoxLayout()
        combo_layout.setContentsMargins(0, 0, 0, 10)

        label_font = QtGui.QFont("微软雅黑", 8)
        comboBox_font = QtGui.QFont("微软雅黑", 9)

        # 叫地主策略
        layout1 = QtWidgets.QVBoxLayout()
        label1 = QtWidgets.QLabel("叫地主策略：")
        label1.setFont(label_font)
        label1.setAlignment(QtCore.Qt.AlignLeft)

        self.cbBid = QtWidgets.QComboBox()
        self.cbBid.addItem('稳健+ (0.75)', 0.75)
        self.cbBid.addItem('稳健  (0.7)', 0.7)
        self.cbBid.addItem('均衡+ (0.65)', 0.65)
        self.cbBid.addItem('均衡  (0.6)', 0.6)
        self.cbBid.addItem('均衡- (0.55)', 0.55)
        self.cbBid.addItem('进取  (0.5)', 0.5)
        self.cbBid.addItem('进取+ (0.45)', 0.45)
        self.cbBid.setFont(comboBox_font)
        self.cbBid.setFixedWidth(180)
        self.cbBid.setCurrentIndex(3)
        self.cbBid.currentIndexChanged.connect(self.handle_cbBid_selection_changed)

        layout1.addWidget(label1)
        layout1.addWidget(self.cbBid)
        layout1.setAlignment(QtCore.Qt.AlignLeft)
        combo_layout.addLayout(layout1)

        # 加倍x2策略
        layout2 = QtWidgets.QVBoxLayout()
        label2 = QtWidgets.QLabel("加倍 x2 策略：")
        label2.setFont(label_font)
        label2.setAlignment(QtCore.Qt.AlignLeft)

        self.cbRedoubleX2 = QtWidgets.QComboBox()
        self.cbRedoubleX2.addItem('稳健+ (0.8)', 0.8)
        self.cbRedoubleX2.addItem('稳健  (0.75)', 0.75)
        self.cbRedoubleX2.addItem('均衡+ (0.7)', 0.7)
        self.cbRedoubleX2.addItem('均衡  (0.65)', 0.65)
        self.cbRedoubleX2.addItem('均衡- (0.6)', 0.6)
        self.cbRedoubleX2.addItem('进取  (0.55)', 0.55)
        self.cbRedoubleX2.addItem('进取+ (0.5)', 0.5)
        self.cbRedoubleX2.setFont(comboBox_font)
        self.cbRedoubleX2.setFixedWidth(180)
        self.cbRedoubleX2.setCurrentIndex(3)
        self.cbRedoubleX2.currentIndexChanged.connect(self.handle_cbRedoubleX2_selection_changed)

        layout2.addWidget(label2)
        layout2.addWidget(self.cbRedoubleX2)
        layout2.setAlignment(QtCore.Qt.AlignLeft)
        combo_layout.addLayout(layout2)

        # 加倍x4策略
        layout3 = QtWidgets.QVBoxLayout()
        label3 = QtWidgets.QLabel("加倍 x4 策略：")
        label3.setFont(label_font)
        label3.setAlignment(QtCore.Qt.AlignLeft)

        self.cbRedoubleX4 = QtWidgets.QComboBox()
        self.cbRedoubleX4.addItem('稳健+ (0.85)', 0.85)
        self.cbRedoubleX4.addItem('稳健  (0.8)', 0.8)
        self.cbRedoubleX4.addItem('均衡+ (0.75)', 0.75)
        self.cbRedoubleX4.addItem('均衡  (0.7)', 0.7)
        self.cbRedoubleX4.addItem('均衡- (0.65)', 0.65)
        self.cbRedoubleX4.addItem('进取  (0.6)', 0.6)
        self.cbRedoubleX4.addItem('进取+ (0.55)', 0.55)
        self.cbRedoubleX4.setFont(comboBox_font)
        self.cbRedoubleX4.setFixedWidth(180)
        self.cbRedoubleX4.setCurrentIndex(3)
        self.cbRedoubleX4.currentIndexChanged.connect(self.handle_cbRedoubleX4_selection_changed)

        layout3.addWidget(label3)
        layout3.addWidget(self.cbRedoubleX4)
        layout3.setAlignment(QtCore.Qt.AlignLeft)
        combo_layout.addLayout(layout3)

        # 明牌策略
        layout4 = QtWidgets.QVBoxLayout()
        label4 = QtWidgets.QLabel("明牌策略：")
        label4.setFont(label_font)
        label4.setAlignment(QtCore.Qt.AlignLeft)

        self.cbMingpai = QtWidgets.QComboBox()
        self.cbMingpai.addItem('稳健+ (1.1)', 1.1)
        self.cbMingpai.addItem('稳健  (0.99)', 0.99)
        self.cbMingpai.addItem('均衡+ (0.97)', 0.97)
        self.cbMingpai.addItem('均衡  (0.95)', 0.95)
        self.cbMingpai.addItem('均衡- (0.93)', 0.93)
        self.cbMingpai.addItem('进取  (0.91)', 0.91)
        self.cbMingpai.addItem('进取+ (0.89)', 0.89)
        self.cbMingpai.setFont(comboBox_font)
        self.cbMingpai.setFixedWidth(180)
        self.cbMingpai.setCurrentIndex(1)
        self.cbMingpai.currentIndexChanged.connect(self.handle_cbMingpai_selection_changed)

        layout4.addWidget(label4)
        layout4.addWidget(self.cbMingpai)
        layout4.setAlignment(QtCore.Qt.AlignLeft)
        combo_layout.addLayout(layout4)

        combo_layout.addStretch()
        self.main_layout.addLayout(combo_layout)

    def create_card_counter_table(self):
        # 创建 QTableWidget
        self.cardCounterTable = QtWidgets.QTableWidget(self)
        self.cardCounterTable.setGeometry(0, 0, 750, 100)  # 设置表格位置和大小
        self.cardCounterTable.setFixedHeight(102)
        self.cardCounterTable.setRowCount(2)
        self.cardCounterTable.setColumnCount(15)

        # 隐藏表头和行头
        self.cardCounterTable.verticalHeader().setVisible(False)
        self.cardCounterTable.horizontalHeader().setVisible(False)

        # 去除网格线
        self.cardCounterTable.setShowGrid(False)

        # 设置行高和列宽
        for row in range(2):
            self.cardCounterTable.setRowHeight(row, 50)
        for col in range(15):
            self.cardCounterTable.setColumnWidth(col, 50)

        # 设置卡牌名称和数量
        card_names = ['大', '小', '2', 'A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3']
        card_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        font1 = QtGui.QFont("微软雅黑", 9, QtGui.QFont.Bold)
        font2 = QtGui.QFont("微软雅黑", 10, QtGui.QFont.Bold)

        for col, name in enumerate(card_names):
            item = QtWidgets.QTableWidgetItem(name)
            item.setFont(font1)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            item.setBackground(QtGui.QColor('#7c7c7c'))
            item.setForeground(QtGui.QColor('#202020'))
            self.cardCounterTable.setItem(0, col, item)

        for col, count in enumerate(card_counts):
            item = QtWidgets.QTableWidgetItem(str(count))
            item.setFont(font2)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            item.setBackground(QtGui.QColor('#404040'))
            item.setForeground(QtGui.QColor('#ffffff'))
            self.cardCounterTable.setItem(1, col, item)

        # 设置表格不可滚动
        self.cardCounterTable.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.cardCounterTable.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.cardCounterTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.cardCounterTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        
        self.main_layout.addWidget(self.cardCounterTable)

    def create_label(self):
        label_layout = QtWidgets.QHBoxLayout()
        label_layout.setContentsMargins(0, 0, 0, 10)
        label_layout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        font = QtGui.QFont("微软雅黑", 8)
        label = QtWidgets.QLabel("底牌：")
        label.setFont(font)

        font2 = QtGui.QFont("微软雅黑", 9, QtGui.QFont.Bold)
        self.threeCardsLabel = QtWidgets.QLabel("---")
        self.threeCardsLabel.setFont(font2)

        label_layout.addWidget(label)
        label_layout.addWidget(self.threeCardsLabel)

        label_layout.addStretch()
        
        label2 = QtWidgets.QLabel("我的身份：")
        label2.setFont(font)

        self.myPositionLabel = QtWidgets.QLabel("---")
        self.myPositionLabel.setFont(font2)

        label_layout.addWidget(label2)
        label_layout.addWidget(self.myPositionLabel)

        self.main_layout.addLayout(label_layout)
        self.main_layout.addStretch()

    def create_other_tables(self):
        self.create_predict_info_table()
        self.create_suggestion_table()
        self.create_played_card_table()

        table_layout = QtWidgets.QHBoxLayout()
        table_layout.addWidget(self.predictInfoTable)
        table_layout.addWidget(self.suggestionTable)
        table_layout.addWidget(self.playedCardsTable)

        self.main_layout.addLayout(table_layout)

    def create_predict_info_table(self):
        self.predictInfoTable = QtWidgets.QTableWidget(self)
        self.predictInfoTable.setGeometry(0, 0, 200, 150)
        self.predictInfoTable.setFixedSize(200, 150)
        self.predictInfoTable.setRowCount(4)
        self.predictInfoTable.setColumnCount(2)

        self.predictInfoTable.verticalHeader().setVisible(False)
        self.predictInfoTable.horizontalHeader().setVisible(False)
        self.predictInfoTable.setShowGrid(False)

        for row in range(4):
            self.predictInfoTable.setRowHeight(row, 33)

        self.predictInfoTable.setColumnWidth(0, 118)
        self.predictInfoTable.setColumnWidth(1, 78)

        headers = ['AI 预测', '胜率']
        contents = [
            ('叫地主', '-'),
            ('不叫地主', '-'),
            ('本回合', '-')
        ]

        font_header = QtGui.QFont("微软雅黑", 8)
        font_content = QtGui.QFont("微软雅黑", 8, QtGui.QFont.Bold)

        for col, header in enumerate(headers):
            item = QtWidgets.QTableWidgetItem(header)
            item.setFont(font_header)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.predictInfoTable.setItem(0, col, item)

        for row, (left, right) in enumerate(contents, start=1):
            left_item = QtWidgets.QTableWidgetItem(left)
            left_item.setFont(font_content)
            left_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.predictInfoTable.setItem(row, 0, left_item)

            right_item = QtWidgets.QTableWidgetItem(right)
            right_item.setFont(font_content)
            right_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.predictInfoTable.setItem(row, 1, right_item)

        self.predictInfoTable.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.predictInfoTable.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.predictInfoTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.predictInfoTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        self.predictInfoTable.setStyleSheet("""
            QTableWidget::item {
                background-color: #F3F3F3;
            }
            QTableWidget::item:alternate {
                background-color: #E3E3E3;
            }
        """)

        self.predictInfoTable.setAlternatingRowColors(True)

    def create_suggestion_table(self):
        self.suggestionTable = QtWidgets.QTableWidget(self)
        self.suggestionTable.setGeometry(0, 209, 266, 150)
        self.suggestionTable.setFixedSize(266, 150)
        self.suggestionTable.setRowCount(4)
        self.suggestionTable.setColumnCount(3)

        self.suggestionTable.verticalHeader().setVisible(False)
        self.suggestionTable.horizontalHeader().setVisible(False)
        self.suggestionTable.setShowGrid(False)

        for row in range(4):
            self.suggestionTable.setRowHeight(row, 33)

        self.suggestionTable.setColumnWidth(0, 130)
        self.suggestionTable.setColumnWidth(1, 58)
        self.suggestionTable.setColumnWidth(2, 74)

        headers = ['AI 建议', '模型分', '敌方可压']
        contents = [
            ('-', '-', '-'),
            ('-', '-', '-'),
            ('-', '-', '-')
        ]

        font_header = QtGui.QFont("微软雅黑", 8)
        font_content = QtGui.QFont("微软雅黑", 8, QtGui.QFont.Bold)

        for col, header in enumerate(headers):
            item = QtWidgets.QTableWidgetItem(header)
            item.setFont(font_header)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.suggestionTable.setItem(0, col, item)

        for row, (action, score, risk) in enumerate(contents, start=1):
            for col, value in enumerate((action, score, risk)):
                item = QtWidgets.QTableWidgetItem(value)
                item.setFont(font_content)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.suggestionTable.setItem(row, col, item)

        self.suggestionTable.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.suggestionTable.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.suggestionTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.suggestionTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        self.suggestionTable.setStyleSheet("""
            QTableWidget::item {
                background-color: #F3F3F3;
            }
            QTableWidget::item:alternate {
                background-color: #E3E3E3;
            }
        """)

        self.suggestionTable.setAlternatingRowColors(True)

    def create_played_card_table(self):
        self.playedCardsTable = QtWidgets.QTableWidget(self)
        self.playedCardsTable.setGeometry(0, 484, 266, 150)
        self.playedCardsTable.setFixedSize(266, 150)
        self.playedCardsTable.setRowCount(4)
        self.playedCardsTable.setColumnCount(2)

        self.playedCardsTable.verticalHeader().setVisible(False)
        self.playedCardsTable.horizontalHeader().setVisible(False)
        self.playedCardsTable.setShowGrid(False)

        for row in range(4):
            self.playedCardsTable.setRowHeight(row, 33)
            
        self.playedCardsTable.setColumnWidth(0, 94)
        self.playedCardsTable.setColumnWidth(1, 168)

        headers = ['玩家', '上一次出牌']
        contents = [
            ('上家', '-'),
            ('地主', '-'),
            ('下家', '-')
        ]

        font_header = QtGui.QFont("微软雅黑", 8)
        font_content = QtGui.QFont("微软雅黑", 8, QtGui.QFont.Bold)

        for col, header in enumerate(headers):
            item = QtWidgets.QTableWidgetItem(header)
            item.setFont(font_header)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.playedCardsTable.setItem(0, col, item)

        for row, (left, right) in enumerate(contents, start=1):
            left_item = QtWidgets.QTableWidgetItem(left)
            left_item.setFont(font_content)
            left_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.playedCardsTable.setItem(row, 0, left_item)

            right_item = QtWidgets.QTableWidgetItem(right)
            right_item.setFont(font_content)
            right_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.playedCardsTable.setItem(row, 1, right_item)

        self.playedCardsTable.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.playedCardsTable.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.playedCardsTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.playedCardsTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        self.playedCardsTable.setStyleSheet("""
            QTableWidget::item {
                background-color: #F3F3F3;
            }
            QTableWidget::item:alternate {
                background-color: #E3E3E3;
            }
        """)

        self.playedCardsTable.setAlternatingRowColors(True)

    def create_actions(self):
        right_layout = QtWidgets.QHBoxLayout()
        right_layout.setContentsMargins(0, 10, 0, 0)
        right_layout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        
        font = QtGui.QFont("微软雅黑", 9)

        self.cbMode = QtWidgets.QComboBox()
        if getattr(self.screenHelper.config, 'platform', '') == 'wechat_miniapp':
            self.cbMode.addItem("微信只读模式", AutomaticModeEnum.MANUAL.value)
        else:
            self.cbMode.addItem("全自动模式", AutomaticModeEnum.FULL.value)
            self.cbMode.addItem("半自动模式", AutomaticModeEnum.SEMI.value)
            self.cbMode.addItem("手动模式", AutomaticModeEnum.MANUAL.value)
        self.cbMode.setFont(font)
        self.cbMode.setFixedWidth(140)
        self.cbMode.setFixedHeight(40)
        self.cbMode.currentIndexChanged.connect(self.handle_cbMode_selection_changed)

        self.startBtn = QtWidgets.QPushButton("启动", self)
        self.startBtn.setFont(font)
        self.startBtn.setFixedWidth(120)
        self.startBtn.setFixedHeight(40)
        self.startBtn.clicked.connect(self.handle_startBtn_clicked)

        right_layout.addWidget(self.cbMode)
        right_layout.addWidget(self.startBtn)

        self.main_layout.addLayout(right_layout)
        self.main_layout.addStretch()