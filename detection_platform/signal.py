from PySide2.QtCore import Signal, QObject, QDateTime
from PySide2.QtGui import QPixmap


class MySignals(QObject):

    update_status = Signal(str, str)

    update_processBar = Signal(int)

    update_graph = Signal()

    update_image = Signal(QPixmap, int)

    update_dateTime = Signal(QDateTime)

    update_windowInfo = Signal(str, str)