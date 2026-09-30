from collections import Counter
import ctypes
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from constants import RealCards


DISPLAY_CARD = {
    "D": "大王",
    "X": "小王",
    "T": "10",
}


def display_cards(cards):
    if not cards:
        return "-"
    return " ".join(DISPLAY_CARD.get(card, card) for card in cards)


class HudPanel(QtWidgets.QFrame):
    """Small draggable always-on-top HUD panel.

    Edit mode:
        - accepts mouse input
        - blue border
        - can be dragged anywhere

    Locked mode:
        - ignores mouse input
        - on Windows the native window also receives WS_EX_TRANSPARENT so
          clicks pass through to the game window underneath
    """

    moved = QtCore.pyqtSignal(str, QtCore.QPoint)

    def __init__(self, panel_id, title, size, parent=None):
        super().__init__(parent)
        self.panel_id = panel_id
        self._editing = True
        self._drag_offset = None

        self.setWindowFlags(
            QtCore.Qt.Tool
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setFixedSize(*size)

        self.frame = QtWidgets.QFrame(self)
        self.frame.setObjectName("hudFrame")
        self.frame.setGeometry(self.rect())

        layout = QtWidgets.QVBoxLayout(self.frame)
        layout.setContentsMargins(10, 7, 10, 8)
        layout.setSpacing(3)

        self.titleLabel = QtWidgets.QLabel(title)
        self.titleLabel.setObjectName("hudTitle")
        self.titleLabel.setFont(QtGui.QFont("微软雅黑", 10, QtGui.QFont.Bold))
        self.titleLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.bodyLabel = QtWidgets.QLabel("等待牌局状态")
        self.bodyLabel.setObjectName("hudBody")
        self.bodyLabel.setFont(QtGui.QFont("微软雅黑", 10, QtGui.QFont.Bold))
        self.bodyLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.bodyLabel.setWordWrap(True)

        # Let the top-level panel receive drag events instead of the labels.
        self.frame.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.titleLabel.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.bodyLabel.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        layout.addWidget(self.titleLabel)
        layout.addWidget(self.bodyLabel, 1)

        self._apply_style()

    def set_title(self, text):
        self.titleLabel.setText(text)

    def set_body(self, text):
        self.bodyLabel.setText(text or "-")

    def _apply_style(self):
        border = "#4da3ff" if self._editing else "rgba(255,255,255,90)"
        self.setStyleSheet(
            f"""
            QFrame#hudFrame {{
                background-color: rgba(16, 18, 24, 218);
                border: 2px solid {border};
                border-radius: 8px;
            }}
            QLabel#hudTitle {{
                color: #76b9ff;
                background: transparent;
            }}
            QLabel#hudBody {{
                color: #ffffff;
                background: transparent;
            }}
            """
        )

    def set_edit_mode(self, editing):
        self._editing = bool(editing)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, not self._editing)
        self._apply_native_click_through(not self._editing)
        self._apply_style()

    def _apply_native_click_through(self, enabled):
        if sys.platform != "win32" or not self.isVisible():
            return

        try:
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            gwl_exstyle = -20
            ws_ex_transparent = 0x00000020
            ws_ex_layered = 0x00080000
            ws_ex_noactivate = 0x08000000

            style = user32.GetWindowLongW(hwnd, gwl_exstyle)
            style |= ws_ex_layered | ws_ex_noactivate
            if enabled:
                style |= ws_ex_transparent
            else:
                style &= ~ws_ex_transparent
            user32.SetWindowLongW(hwnd, gwl_exstyle, style)
        except Exception:
            # Qt-level mouse transparency still provides a safe fallback.
            pass

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_native_click_through(not self._editing)

    def mousePressEvent(self, event):
        if self._editing and event.button() == QtCore.Qt.LeftButton:
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._editing
            and self._drag_offset is not None
            and event.buttons() & QtCore.Qt.LeftButton
        ):
            self.move(event.globalPos() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._editing and event.button() == QtCore.Qt.LeftButton:
            self._drag_offset = None
            self.moved.emit(self.panel_id, self.pos())
            event.accept()
            return
        super().mouseReleaseEvent(event)


class HudOverlayManager(QtCore.QObject):
    """Owns the four detachable HUD modules and keeps their state persistent."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QtCore.QSettings("DouZeroAssistant", "DetachableHUD")
        configured = self.settings.value("configured", False, type=bool)
        self.edit_mode = not configured
        self.visible = self.settings.value("visible", True, type=bool)

        self.panels = {
            "left": HudPanel("left", "左家推测", (340, 128)),
            "right": HudPanel("right", "右家推测", (340, 128)),
            "suggestion": HudPanel("suggestion", "AI 建议", (430, 138)),
            "counter": HudPanel("counter", "记牌器", (720, 82)),
        }

        for panel in self.panels.values():
            panel.moved.connect(self._save_panel_position)
            panel.set_edit_mode(self.edit_mode)

        self._place_panels()
        if self.visible:
            self.show()

    def _screen_geometry(self):
        app = QtWidgets.QApplication.instance()
        screen = app.primaryScreen() if app else None
        if screen is None:
            return QtCore.QRect(0, 0, 1920, 1080)
        return screen.availableGeometry()

    def _default_positions(self):
        g = self._screen_geometry()
        return {
            "left": QtCore.QPoint(g.left() + 20, g.top() + 250),
            "right": QtCore.QPoint(g.right() - 360, g.top() + 250),
            "suggestion": QtCore.QPoint(
                g.center().x() - 215, g.top() + 80
            ),
            "counter": QtCore.QPoint(
                g.center().x() - 360, g.bottom() - 120
            ),
        }

    def _place_panels(self):
        defaults = self._default_positions()
        for panel_id, panel in self.panels.items():
            saved = self.settings.value(f"panels/{panel_id}/pos")
            if isinstance(saved, QtCore.QPoint):
                panel.move(saved)
            else:
                panel.move(defaults[panel_id])

    def _save_panel_position(self, panel_id, pos):
        self.settings.setValue(f"panels/{panel_id}/pos", pos)
        self.settings.sync()

    def set_edit_mode(self, editing):
        self.edit_mode = bool(editing)
        for panel in self.panels.values():
            panel.set_edit_mode(self.edit_mode)
        if not self.edit_mode:
            self.settings.setValue("configured", True)
        self.settings.sync()

    def toggle_edit_mode(self):
        self.set_edit_mode(not self.edit_mode)
        return self.edit_mode

    def set_visible(self, visible):
        self.visible = bool(visible)
        self.settings.setValue("visible", self.visible)
        for panel in self.panels.values():
            if self.visible:
                panel.show()
                panel.raise_()
                panel.set_edit_mode(self.edit_mode)
            else:
                panel.hide()
        self.settings.sync()

    def toggle_visible(self):
        self.set_visible(not self.visible)
        return self.visible

    def reset_positions(self):
        defaults = self._default_positions()
        for panel_id, panel in self.panels.items():
            panel.move(defaults[panel_id])
            self.settings.setValue(
                f"panels/{panel_id}/pos", defaults[panel_id]
            )
        self.settings.sync()

    def show(self):
        self.set_visible(True)

    def close(self):
        for panel in self.panels.values():
            panel.close()

    def update_counter(self, remaining_cards):
        if not isinstance(remaining_cards, str):
            self.panels["counter"].set_body("等待记牌数据")
            return

        counts = Counter(remaining_cards)
        short_names = {
            "D": "大",
            "X": "小",
            "T": "10",
        }
        names = [short_names.get(card, card) for card in RealCards]
        values = [str(counts.get(card, 0)) for card in RealCards]

        # Use a tiny HTML table so every count sits directly below its rank.
        # This is much easier to scan during a 30-second turn than a single row.
        name_cells = "".join(
            f'<td align="center"><b>{name}</b></td>' for name in names
        )
        value_cells = "".join(
            f'<td align="center">{value}</td>' for value in values
        )
        html = (
            '<table width="100%" cellspacing="0" cellpadding="0">'
            f"<tr>{name_cells}</tr>"
            f"<tr>{value_cells}</tr>"
            "</table>"
        )
        self.panels["counter"].set_body(html)

    def update_suggestion(self, result):
        panel = self.panels["suggestion"]
        if (
            isinstance(result, list)
            and result
            and isinstance(result[0], (list, tuple))
            and result[0]
            and result[0][0] == "__PAUSED__"
        ):
            panel.set_title("AI 建议 · 状态暂停")
            panel.set_body("等待牌局历史与剩余张数重新同步")
            return

        panel.set_title("AI 建议")
        if not result or not isinstance(result, list):
            panel.set_body("等待我的回合")
            return

        rows = []
        for i, data in enumerate(result[:3], start=1):
            action = data[0] if len(data) > 0 else "-"
            score = data[1] if len(data) > 1 else "-"
            total = data[2] if len(data) > 2 else "-"
            if action == "Pass":
                action_text = "不出"
            elif action in ("-", None):
                action_text = "-"
            else:
                action_text = display_cards(action)
            rows.append(f"{i}. {action_text}   分 {score}   敌可压 {total}")
        panel.set_body("\n".join(rows))

    @staticmethod
    def _role_name(position):
        return {
            "landlord": "地主",
            "landlord_up": "农民",
            "landlord_down": "农民",
        }.get(position, "")

    @staticmethod
    def _format_player_body(data):
        if not data:
            return "等待推牌数据"

        card_stats = data.get("cards", {})
        ranked = []
        for card in RealCards:
            stats = card_stats.get(card, {})
            p = float(stats.get("one_plus", 0.0) or 0.0)
            if p >= 0.20:
                ranked.append((p, card))
        ranked.sort(reverse=True)

        high = "  ".join(
            f"{DISPLAY_CARD.get(card, card)} {p:.0%}"
            for p, card in ranked[:7]
        )
        if not high:
            high = "暂无明显高概率单牌"

        top_hands = data.get("top_sampled_hands", [])
        if top_hands:
            top = top_hands[0]
            hand_text = display_cards(top.get("hand", ""))
            top_line = f"Top组合 {top.get('probability', 0):.1%}: {hand_text}"
        else:
            top_line = "Top组合：-"

        return (
            f"{top_line}\n"
            f"高概率：{high}\n"
            f"炸弹 {data.get('any_bomb', 0):.0%} · 王炸 {data.get('rocket', 0):.0%}"
        )

    def update_inference(self, result, side_map, observed_remaining):
        players = result.get("players", {}) if isinstance(result, dict) else {}
        for side in ("left", "right"):
            panel = self.panels[side]
            position = side_map.get(side) if isinstance(side_map, dict) else None
            data = players.get(position) if position else None

            count = None
            observed = False
            desync = False
            info = (
                observed_remaining.get(position, {})
                if position and isinstance(observed_remaining, dict)
                else {}
            )
            if isinstance(info, dict):
                count = info.get("count")
                observed = bool(info.get("observed"))
                desync = bool(info.get("desync"))
            if count is None and data:
                count = data.get("remaining_count")

            side_name = "左家" if side == "left" else "右家"
            role = self._role_name(position)
            count_text = "-" if count is None else str(count)
            if observed:
                count_text += "张"
            if desync:
                count_text += "!"

            title = f"{side_name}"
            if role:
                title += f" · {role}"
            title += f" · {count_text}"
            panel.set_title(title)
            panel.set_body(self._format_player_body(data))
