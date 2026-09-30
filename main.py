import asyncio
import os
import signal
import sys


def _configure_qt_plugin_paths():
    """Make PyQt5 platform plugins discoverable inside the Windows venv."""
    plugin_root = os.path.join(
        sys.prefix, "Lib", "site-packages", "PyQt5", "Qt5", "plugins"
    )
    platform_root = os.path.join(plugin_root, "platforms")

    if os.path.isdir(plugin_root):
        os.environ.setdefault("QT_PLUGIN_PATH", plugin_root)
    if os.path.isdir(platform_root):
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", platform_root)


_configure_qt_plugin_paths()

import qasync
from PyQt5 import QtCore

from main_window import MainWindow


if __name__ == "__main__":
    app = qasync.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    mainWindow = MainWindow()
    mainWindow.show()

    # Keep shutdown state mutable because this code runs at module scope.
    shutdown_state = {"active": False}

    def shutdown(*_):
        if shutdown_state["active"]:
            return
        shutdown_state["active"] = True
        print("正在退出程序...")
        try:
            mainWindow.close()
        finally:
            app.quit()

    # Make Ctrl+C work reliably while Qt owns the foreground event loop.
    signal.signal(signal.SIGINT, shutdown)

    # On Windows, periodically hand control back to Python so SIGINT is
    # processed promptly even when there is no other Qt activity.
    signal_timer = QtCore.QTimer()
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start(200)

    app.aboutToQuit.connect(loop.stop)

    try:
        with loop:
            loop.run_forever()
    except KeyboardInterrupt:
        shutdown()
