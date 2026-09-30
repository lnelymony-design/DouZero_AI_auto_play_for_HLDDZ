import asyncio
import os
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

from main_window import MainWindow

if __name__ == '__main__':
    app = qasync.QApplication(sys.argv)

    mainWindow = MainWindow()
    mainWindow.show()

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    with loop:
        loop.run_forever()

    # Execute application
    sys.exit(app.exec_())
