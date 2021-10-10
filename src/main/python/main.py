from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QMainWindow, QDialogButtonBox, QVBoxLayout, QLabel, QDialog, QFileDialog
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
import sys
import os
import cv2
import numpy as np
import time

class ImageMethods:
    @staticmethod
    # Dummy Actions applied to image!
    def gaussian_blur(image):
        return cv2.GaussianBlur(image, (37,37), cv2.BORDER_DEFAULT)

    @staticmethod
    def gaussian_noise(image, seed=10):
        noise_matrix = np.random.normal(2, 50, size=image.shape)
        image = image+noise_matrix
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

class MethodType:
    GAUSSIAN_NOISE = 0
    GAUSSIAN_BLUR = 1

class MediaType:
    IMAGE = 0
    VIDEO_FILE = 1
    VIDEO_STREAM = 2

class PresetOption:
    TITLE = ''
    MEDIA_TYPE = ''
    MEDIA_PATH = ''
    MEDIA_OUT_PATH = ''
    METHOD = ''
    def __init__(self, title, media_type, media_path, media_out_path, method) -> None:
        self.TITLE = title
        self.MEDIA_PATH = media_path
        self.MEDIA_TYPE = media_type
        self.MEDIA_OUT_PATH = media_out_path
        self.METHOD = method

class ThreadWorker(QThread):
    #change_pixmap_signal = pyqtSignal(np.ndarray)
    change_pixmap_signal = pyqtSignal(tuple)

    def __init__(self,  media_type, media_path, media_out_path, method):
        super().__init__()
        self.media_type = media_type
        self.media_path = media_path
        self.media_out_path = media_out_path
        self.method = method

    def run(self):
        self._run_flag = True

        if self.media_type == MediaType.IMAGE:
            t1 = time.time()
            img = cv2.imread(self.media_path)
            assert type(img) != type(None), "Image cannot be read: %s"%self.media_path
            if self.method == MethodType.GAUSSIAN_NOISE:
                img = ImageMethods.gaussian_noise(img)
            elif self.method == MethodType.GAUSSIAN_BLUR:
                img = ImageMethods.gaussian_blur(img)
            cv2.imwrite(self.media_out_path, img)
            self.change_pixmap_signal.emit((img, t1))

        elif self.media_type == MediaType.VIDEO_FILE:
            videoIn = cv2.VideoCapture(self.media_path)
            videoOut = None

            while(self._run_flag):
                t1 = time.time()
                ret, frame = videoIn.read()
                if not ret:
                    break

                if type(videoOut) == type(None):
                    videoOut = cv2.VideoWriter(self.media_out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame.shape[1], frame.shape[0]))

                if self.method == MethodType.GAUSSIAN_NOISE:
                    frame = ImageMethods.gaussian_noise(frame)
                elif self.method == MethodType.GAUSSIAN_BLUR:
                    frame = ImageMethods.gaussian_blur(frame)
                videoOut.write(frame)

                self.change_pixmap_signal.emit((frame, t1))
            videoIn.release()
            videoOut.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self, appContext):
        super(MainWindow, self).__init__()
        path = appContext.get_resource('mainwindow.ui')
        self.imageFolderExample = appContext.get_resource('example_images')
        self.videoFolderExample = appContext.get_resource('example_videos')
        self.base = appContext.get_resource("")
        self.__presetSetup__()

        # Set dummy variable for main worker thread:
        self.worker = None
    
        uic.loadUi(path, self)
        self.setWindowTitle("Demo Application")
        # Set up combo box with presets and custom:
        for preset in self.presets:
            self.presetComboBox.addItem(preset.TITLE)
        self.presetComboBox.addItem("Custom")
        self.inputPath.setText(self.presets[0].MEDIA_PATH)
        self.outputPath.setText(self.presets[0].MEDIA_OUT_PATH)
        self.__toggle_all_controls__(False)
        self.presetComboBox.currentTextChanged.connect(self.__handlePresetBox__)
        
        # connect event to run button:
        self.runButton.clicked.connect(self.__launch__)
        self.imageRadio.setChecked(True)

        # settings for pixMap label:
        self.disply_width = self.pixMap.frameGeometry().width()
        self.display_height = self.pixMap.frameGeometry().height()

        # file browser button actions:
        self.inputBrowse.clicked.connect(lambda: self.__openFileBrowser__(self.imageFolderExample, self.inputPath))
        self.outputBrowse.clicked.connect(lambda: self.__openFileBrowser__(self.base, self.outputPath))
        
        self.show()

    def __presetSetup__(self):
        self.presets = [
            PresetOption(
                'Image Demo #1',
                MediaType.IMAGE,
                os.path.join(self.imageFolderExample, 'selfie.jpg'),
                os.path.join(self.imageFolderExample, 'out_selfie.jpg'),
                MethodType.GAUSSIAN_NOISE
            ),
            PresetOption(
                'Video Demo #1',
                MediaType.VIDEO_FILE, 
                os.path.join(self.videoFolderExample, 'people_walking.mp4'),
                os.path.join(self.videoFolderExample, 'out_people_walking.mp4'),
                MethodType.GAUSSIAN_BLUR
            )
        ]

    def __handlePresetBox__(self):
        comboBoxLength = self.presetComboBox.count()
        currentI = self.presetComboBox.currentIndex()
        if currentI != comboBoxLength-1:
            # is an actual preset
            _preset = self.presets[currentI]
            self.inputPath.setText(_preset.MEDIA_PATH)
            self.outputPath.setText(_preset.MEDIA_OUT_PATH)
            self.__toggle_all_controls__(False)
        else:
            self.__toggle_all_controls__(True)
    
    def __toggle_all_controls__(self, value):
        self.inputPath.setEnabled(value)
        self.inputBrowse.setEnabled(value)
        self.outputPath.setEnabled(value)
        self.outputBrowse.setEnabled(value)
        self.imageRadio.setEnabled(value)
        self.videoRadio.setEnabled(value)
        #self.runButton.setEnabled(value)

    @pyqtSlot(tuple)
    def __updateImage__(self, frame_tuple):
        frame, t1 = frame_tuple
        qtFrame = self.convertCV2QT(frame)
        self.pixMap.setPixmap(qtFrame)
        t2 = time.time()
        self.fpsLabelStat.setText("FPS: %0.3f"%(1/(t2-t1)))

    def __launch__(self):
        if self.worker is not None and self.worker.isRunning():
            # This is where the button will reset the worker from whatever it was working on:
            # This does not kill the thread variable, only the thread allocation
            self.worker.stop()
            self.runButton.setText("RUN")
        else:
            comboBoxLength = self.presetComboBox.count()
            currentI = self.presetComboBox.currentIndex()
            if currentI != comboBoxLength-1:
                # presets:
                media_type = self.presets[currentI].MEDIA_TYPE
                media_path = self.presets[currentI].MEDIA_PATH
                media_out_path = self.presets[currentI].MEDIA_OUT_PATH
                method = self.presets[currentI].METHOD
            else:
                media_type = self.__determineCustomType__()
                media_path = self.inputPath.text()
                media_out_path = self.outputPath.text()
                method = MethodType.GAUSSIAN_NOISE # hardset for now
                
            if media_type == MediaType.VIDEO_FILE:
                # Change button to "stop" mode:
                self.runButton.setText("STOP VIDEO")

            # Set up another thread for Video/Image I/O:
            self.worker = ThreadWorker(media_type, media_path, media_out_path, method)
            self.worker.change_pixmap_signal.connect(self.__updateImage__)
            self.worker.start()

    def __determineCustomType__(self):
        if self.imageRadio.isChecked():
            # Simple error checking:
            _test = cv2.imread(self.inputPath.text())
            if type(_test) == type(None):
                self.__sendDialogMessage__("%s is not a valid image file!"%self.inputPath.text())
            return MediaType.IMAGE
        elif self.videoRadio.isChecked():
            videoIn = cv2.VideoCapture(self.inputPath.text())
            ret,_ = videoIn.read()
            if not ret:
                self.__sendDialogMessage__("%s is not a valid video file!"%self.inputPath.text())
            videoIn.release()
            return MediaType.VIDEO_FILE

    def __openFileBrowser__(self, default_path, textLine):
        root_name = QFileDialog.getOpenFileName(
            self, "Select Folder", default_path)
        if len(root_name[0]) > 0:
            textLine.setText(root_name[0])

    def convertCV2QT(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def __sendDialogMessage__(self, message):
        diag = CustomDialog(message)
        if diag.exec():
            print(message)
        else:
            print(message)

class CustomDialog(QDialog):
    def __init__(self, str_message, window_title="Error Detected!"):
        super().__init__()

        self.setWindowTitle(window_title)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel(str_message)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

if __name__ == '__main__':
    print("Starting application...")
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    window = MainWindow(appctxt)
    exit_code = appctxt.app.exec_()
    sys.exit(exit_code)