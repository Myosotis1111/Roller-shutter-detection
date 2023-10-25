import functools
import io
import os
import shutil
import tkinter as tk
from threading import Thread
from tkinter import filedialog, messagebox
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PySide2.QtCore import QSize
from PySide2.QtGui import QImage, QPixmap, Qt, QMouseEvent
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QMessageBox, QDialog, QLabel, QFrame
from model import Model


class MainView:

    def __init__(self):

        self.m = Model()

        # # The following three lines can be enabled if you want to use a fixed window location of each image
        # self.m.txt_mode = "fixed"
        # # if txt_mode = "fixed", cropping of images will follow the same text file for each position.
        # # else by default, cropping of images will follow the corresponding text file of each image.
        # # the path of text file including window position for dingcam02.
        # self.m.cam_dingcam02_txt_path = "dingcam02.txt"
        # # the path of text file including window position for dingcam03.
        # self.m.cam_dingcam03_txt_path = "dingcam03.txt"

        self.ms = self.m.ms
        self.ms.update_status.connect(self.update_status)
        self.ms.update_processBar.connect(self.update_progress)
        self.ms.update_graph.connect(self.update_graph)
        self.ms.update_image.connect(self.update_image)
        self.ms.update_dateTime.connect(self.update_dateTime)
        self.ms.update_windowInfo.connect(self.update_windowInfo)

        self.ui = QUiLoader().load('Detection Platform.ui')
        super().__init__()
        self.ui.comboBox.addItem("open")
        self.ui.comboBox.addItem("closed")
        self.ui.comboBox.addItem("blocked")
        self.ui.comboBox.addItem("not_detected")
        self.ui.comboBox.setCurrentIndex(3)
        self.ui.uploadButton.clicked.connect(self.on_uploadButton_clicked)
        self.ui.detectButton.clicked.connect(self.on_detectButton_clicked)
        self.ui.localizeButton.clicked.connect(self.on_localizeButton_clicked)
        self.ui.saveButton.clicked.connect(self.on_saveButton_clicked)
        self.ui.exportButton.clicked.connect(self.on_exportButton_clicked)

        # define and link button instances
        for i in range(1, 28):
            button_name = f"upper_{i}_Button"
            window_no = f"upper_{i}"
            button = getattr(self.ui, button_name)
            button.clicked.connect(functools.partial(self.on_windowButton_clicked, window_no, button_name))

        for i in range(1, 28):
            button_name = f"middle_{i}_Button"
            window_no = f"middle_{i}"
            button = getattr(self.ui, button_name)
            button.clicked.connect(functools.partial(self.on_windowButton_clicked, window_no, button_name))

        for i in range(1, 19):
            button_name = f"lower_{i}_Button"
            window_no = f"lower_{i}"
            button = getattr(self.ui, button_name)
            button.clicked.connect(functools.partial(self.on_windowButton_clicked, window_no, button_name))

        # clear previous results
        result_folder = "images/result"
        if os.path.exists(result_folder):
            shutil.rmtree(result_folder)
        os.makedirs(result_folder)

    def on_uploadButton_clicked(self):

        # Initialize charts and text
        empty_image = QPixmap(QSize(0, 0))
        empty_image.fill(Qt.transparent)
        self.ui.result_img_1.setPixmap(empty_image)
        self.ui.result_img_2.setPixmap(empty_image)
        self.ui.result_img_3.setPixmap(empty_image)
        self.ui.label_line.setPixmap(empty_image)
        self.ui.label_pie.setPixmap(empty_image)
        self.ui.windowText.setText("")
        self.ui.comboBox.setCurrentIndex(3)

        # Reset labels
        self.ui.label0.clear()
        self.ui.label1.clear()
        self.ui.label2.clear()

        # Initialize corresponding buttons of each window
        for i in range(1, 28):
            button_name = f"upper_{i}_Button"
            button = getattr(self.ui, button_name)
            button.setStyleSheet("background-color: #FFFFFF;")

        for i in range(1, 28):
            button_name = f"middle_{i}_Button"
            button = getattr(self.ui, button_name)
            button.setStyleSheet("background-color: #FFFFFF;")

        for i in range(1, 19):
            button_name = f"lower_{i}_Button"
            button = getattr(self.ui, button_name)
            button.setStyleSheet("background-color: #FFFFFF;")

        # clear previous tmp images
        tmp_folder = "images/tmp"
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder)

        self.m.upload()

    def on_detectButton_clicked(self):

        self.ui.progressBar.reset()

        # clear previous results
        result_folder = "images/result"
        if os.path.exists(result_folder):
            shutil.rmtree(result_folder)
        os.makedirs(result_folder)

        # Set confidence threshold, default = 0.25
        try:
            set_conf = float(self.ui.confidenceEdit.toPlainText())
            if 0 <= set_conf <= 1:
                self.m.conf_thres = set_conf
            else:
                QMessageBox.warning(None, "Invalid Threshold", "Confidence threshold must be a number between 0 and 1.")
                return
        except ValueError:
            if not self.ui.confidenceEdit.toPlainText():
                self.m.conf_thres = 0.25
            else:
                QMessageBox.warning(None, "Invalid Threshold", "Confidence threshold must be a number between 0 and 1.")
                return

        # Set NMS IOU threshold, default = 0.45
        try:
            set_IoU = float(self.ui.IoUEdit.toPlainText())
            if 0 <= set_IoU <= 1:
                self.m.iou_thres = set_IoU
            else:
                QMessageBox.warning(None, "Invalid Threshold",
                                    "IoU threshold must be a number between 0 and 1.")
                return
        except ValueError:
            if not self.ui.IoUEdit.toPlainText():
                self.m.iou_thres = 0.45
            else:
                QMessageBox.warning(None, "Invalid Threshold",
                                    "IoU threshold must be a number between 0 and 1.")
                return

        def detectThread():

            self.m.detect()

        t = Thread(target=detectThread)
        t.start()

    def on_exportButton_clicked(self):

        self.m.export()
        title = self.ui.dateTimeEdit.text().replace('/', '-').replace(':', ';')
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.asksaveasfilename(defaultextension='.xlsx', initialfile='{}.xlsx'.format(title))

        if file_path:
            self.m.workbook.save(file_path)
            messagebox.showinfo('Export Successful', 'File has been exported successfully!')

    def on_windowButton_clicked(self, window_no, button):

        # windowButtons refers to the 72 buttons corresponding to 72 windows of the building
        # Initialize the display interface
        self.ui.windowText.clear()
        self.ui.result_img_1.clear()
        self.ui.result_img_2.clear()
        self.ui.result_img_3.clear()

        status = self.m.windows[window_no].status
        if status == "open":
            self.ui.comboBox.setCurrentIndex(0)
        elif status == "closed":
            self.ui.comboBox.setCurrentIndex(1)
        elif status == "blocked":
            self.ui.comboBox.setCurrentIndex(2)
        elif status == "not_detected":
            self.ui.comboBox.setCurrentIndex(3)

        folder_path = "images/result/"
        # Search for files starting with "{window_no}_result"
        target_prefix = f"{window_no}_result"
        i = 0
        for file_name in os.listdir(folder_path):
            if file_name.startswith(target_prefix):
                # Construct the complete path of the file
                file_path = os.path.join(folder_path, file_name)
                # Read the image and display it on the corresponding QLabel
                im = cv2.imread(file_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                height, width, channels = im.shape
                bytesPerLine = channels * width
                qimage = QImage(im.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                if i == 0:
                    self.ui.result_img_1.setPixmap(pixmap)
                    self.ui.result_img_1.setScaledContents(True)
                elif i == 1:
                    self.ui.result_img_2.setPixmap(pixmap)
                    self.ui.result_img_2.setScaledContents(True)
                elif i == 2:
                    self.ui.result_img_3.setPixmap(pixmap)
                    self.ui.result_img_3.setScaledContents(True)
                # Only display the first 3 images
                if i >= 2:
                    break
                i += 1

        if self.m.windows[window_no].status == "closed":
            self.ui.windowText.setText(f"{window_no} is closed. ({self.m.windows[window_no].doc:.2f}%).")
        if self.m.windows[window_no].status == "open":
            self.ui.windowText.setText(f"{window_no} is open.")
        if self.m.windows[window_no].status == "blocked":
            self.ui.windowText.setText(
                f"{window_no} blocked. ({self.m.windows[window_no].doc_min:.2f}% to "
                f"{self.m.windows[window_no].doc_max:.2f}%)")
        if self.m.windows[window_no].status == "not_detected":
            self.ui.windowText.setText(f"{window_no} has not been detected yet.")

    def on_localizeButton_clicked(self):

        text = self.ui.windowText.toPlainText()
        window_no = text.split()[0]
        current_tab = self.ui.tabWidget_2.currentIndex()

        if self.ui.comboBox.currentIndex() == 1:  # If chosen status is "closed"

            # Get the path of the currently selected image
            if current_tab == 0:
                pixmap = self.ui.result_img_1.pixmap()
            elif current_tab == 1:
                pixmap = self.ui.result_img_2.pixmap()
            else:
                pixmap = self.ui.result_img_3.pixmap()

            dialog = LocalizeClosedView(pixmap)
            if dialog.exec_() == QDialog.Accepted:
                shutter_pos = dialog.shutter_pos
                self.ui.windowText.setText(f"{window_no} is closed. ({shutter_pos / 6.4:.2f}%).")

        if self.ui.comboBox.currentIndex() == 2:  # If chosen status is "blocked"

            if current_tab == 0:
                pixmap = self.ui.result_img_1.pixmap()
            elif current_tab == 1:
                pixmap = self.ui.result_img_2.pixmap()
            else:
                pixmap = self.ui.result_img_3.pixmap()

            dialog = LocalizeBlockedView(pixmap)
            if dialog.exec_() == QDialog.Accepted:
                shutter_pos_max = dialog.shutter_pos_max
                shutter_pos_min = dialog.shutter_pos_min
                self.ui.windowText.setText(
                    f"{window_no} blocked. ({shutter_pos_min / 6.4:.2f}% to {shutter_pos_max / 6.4:.2f}%)")

        elif self.ui.comboBox.currentIndex() == 0 or self.ui.comboBox.currentIndex() == 3:
            msg_box = QMessageBox()
            msg_box.setText("You can only localize the shutter when you set its status as closed or blocked!")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.exec_()
            return

    def on_saveButton_clicked(self):

        self.m.window_info = self.ui.windowText.toPlainText()
        self.m.modified_status = self.ui.comboBox.currentText()
        self.m.save()

    def update_status(self, button_no, status):

        button_instance = getattr(self.ui, button_no)
        if status == "closed":
            button_instance.setStyleSheet("background-color: #FFA500;")
        if status == "blocked":
            button_instance.setStyleSheet("background-color: #FF4500;")
        if status == "open":
            button_instance.setStyleSheet("background-color: #87CEEB;")

    def update_progress(self, progress):

        self.ui.progressBar.setValue(progress)

    def update_graph(self):
        # Draw a pie chart
        status_count = {"open": 0, "closed": 0, "blocked": 0, "not_detected": 0}
        for window in self.m.windows.values():
            status_count[window.status] += 1

        # Set the sizes to the count of each status
        sizes = [status_count["open"], status_count["closed"], status_count["blocked"], status_count["not_detected"]]
        labels = ["Open", "Closed", "Blocked", "Not Detected"]
        colors = ["#87CEEB", "#FFA500", "#FF4500", "white"]
        explode = [0.02] * len(sizes)

        fig, ax = plt.subplots()
        wedges, _, autotexts = ax.pie(sizes, labels=labels, colors=colors, explode=explode, startangle=90,
                                      wedgeprops={"width": 0.4, "edgecolor": "black", "linewidth": 2},
                                      textprops={"fontsize": 14, "fontweight": "bold"},
                                      autopct="%1.1f%%")

        # Set font size and style for percentage labels
        for autotext in autotexts:
            autotext.set_fontsize(20)
            autotext.set_bbox(
                dict(facecolor='white', edgecolor='white', alpha=0.7))  # Set background color and transparency

        ax.axis("equal")  # Make the pie chart a perfect circle
        ax.set_title("Window Status", fontweight="bold", fontsize=20, loc="left")
        plt.tight_layout()

        # Save the pie chart as a pixmap
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())
        plt.savefig("images/result/pie_chart.png")

        self.ui.label_pie.setPixmap(pixmap)
        self.ui.label_pie.setScaledContents(True)

        # Select all the Window instances whose status is "closed" and get their doc attribute
        docs = [window.doc for window in self.m.windows.values() if window.status == "closed"]

        # Count the number of times each doc value appears
        bins = np.arange(0, 110, 10)
        hist, _ = np.histogram(docs, bins=bins)

        # Draw a bar chart
        fig, ax = plt.subplots()
        ax.bar(bins[:-1], hist, width=8, align="edge", color="#87CEEB", edgecolor="black", linewidth=2)
        ax.set_xticks(bins)
        ax.set_xlabel("Doc")
        ax.set_ylabel("Count")
        ax.set_ylim(bottom=0)
        ax.set_title("DoC Distribution", fontweight="bold", fontsize=20, loc="left")

        # Add count labels on each bar
        for i, v in enumerate(hist):
            if v != 0:
                ax.bar(bins[i], v, width=8, align="edge", color="#87CEEB", edgecolor="black", linewidth=2)
                ax.text(bins[i] + 4, v - 0.5, str(v), fontweight="bold", ha='center', va='bottom',
                        bbox=dict(facecolor='white', edgecolor='white', alpha=0.7))

        # Save the figure as a pixmap
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())
        plt.savefig("images/result/bar_chart.png")

        # Display the pixmap on the label_line
        self.ui.label_line.setPixmap(pixmap)
        self.ui.label_line.setScaledContents(True)

    def update_image(self, pixmap, i):

        # Display images in GUI
        if i == 0:
            self.ui.label0.setPixmap(pixmap)
            self.ui.label0.setScaledContents(True)
        elif i == 1:
            self.ui.label1.setPixmap(pixmap)
            self.ui.label1.setScaledContents(True)
        elif i == 2:
            self.ui.label2.setPixmap(pixmap)
            self.ui.label2.setScaledContents(True)

    def update_dateTime(self, date_time):

        self.ui.dateTimeEdit.setDateTime(date_time)

    def update_windowInfo(self, window_no, status):

        button_no = f"{window_no}_Button"
        button_instance = getattr(self.ui, button_no)

        # Modify the button appearance according to new status
        if status == "closed":
            self.ui.windowText.setText(f"{window_no} is closed. ({self.m.windows[window_no].doc:.2f}%).")
            button_instance.setStyleSheet("background-color: #FFA500;")
        if status == "blocked":
            self.ui.windowText.setText(
                f"{window_no} blocked. ({self.m.windows[window_no].doc_min:.2f}% to {self.m.windows[window_no].doc_max:.2f}%)")
            button_instance.setStyleSheet("background-color: #FF4500;")
        if status == "open":
            self.ui.windowText.setText(f"{window_no} is open.")
            button_instance.setStyleSheet("background-color: #87CEEB;")
        if status == "not_detected":
            self.ui.windowText.setText(f"{window_no} has not been detected yet.")
            button_instance.setStyleSheet("background-color: #FFFFFF;")


class LocalizeClosedView(QDialog):
    def __init__(self, pixmap):
        super().__init__()
        self.setWindowTitle("Please localize the shutter bottom and press Enter to commit")
        self.setFixedSize(640, 640)

        self.label = QLabel(self)
        self.label.setGeometry(0, 0, self.width(), self.height())

        # Set the original image as the background of the QLabel
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)

        # Register the mousePressEvent event handler function
        self.label.mousePressEvent = self.on_mouse_press_event

        # Register the keyPressEvent event handler function
        self.keyPressEvent = self.on_key_press_event

        # Create and configure a horizontal line
        self.h_line = QFrame(self)
        self.h_line.setFrameStyle(QFrame.HLine)
        self.h_line.setStyleSheet("color: green")
        self.h_line.setFixedWidth(self.width())
        self.h_line.hide()

    def on_mouse_press_event(self, event: QMouseEvent):  # Click to set the new boundary
        self.h_line.setStyleSheet("border: 3px solid green;")
        self.h_line.setFixedHeight(3)
        self.h_line.raise_()
        self.h_line.show()
        self.h_line.move(0, event.pos().y())
        self.shutter_pos = event.pos().y()

    def on_key_press_event(self, event):  # Press Enter to commit
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.accept()


class LocalizeBlockedView(QDialog):
    def __init__(self, pixmap):
        super().__init__()
        self.setWindowTitle("Please click twice to localize the possible shutter top & bottom")
        self.setFixedSize(640, 640)
        self.click_count = 0
        self.shutter_pos_max = None
        self.shutter_pos_min = None

        self.label = QLabel(self)
        self.label.setGeometry(0, 0, self.width(), self.height())

        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)

        self.label.mousePressEvent = self.on_mouse_press_event

        self.h_line = QFrame(self)
        self.h_line.setFrameStyle(QFrame.HLine)
        self.h_line.setStyleSheet("color: green")
        self.h_line.setFixedWidth(self.width())
        self.h_line.hide()

    def on_mouse_press_event(self, event: QMouseEvent):  # Click twice to set the new possible area
        if self.click_count == 0:
            self.h_line.setStyleSheet("border: 3px solid green;")
            self.h_line.setFixedHeight(3)
            self.h_line.raise_()
            self.h_line.show()
            self.h_line.move(0, event.pos().y())
            self.shutter_pos_max = event.pos().y()
            self.shutter_pos_min = event.pos().y()
        else:
            pos_y = event.pos().y()
            self.shutter_pos_max = max(self.shutter_pos_max, pos_y)
            self.shutter_pos_min = min(self.shutter_pos_min, pos_y)

        self.click_count += 1

        if self.click_count == 2:
            # Close the window automatically after clicked twice
            self.accept()
