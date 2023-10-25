"""
-------------------------------------------------
Project Name: Building Management: Machine Vision
File Name: main.py
Author: Xinchen Yang
Last Modified Date: 30/05/2023
Description：used for roller shutter detection in TH Luebeck.
-------------------------------------------------
"""

from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QApplication
from view import MainView

app = QApplication([])
app.setWindowIcon(QIcon("images/logo.png"))
myGUI = MainView()
myGUI.ui.show()
app.exec_()