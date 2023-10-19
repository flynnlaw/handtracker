from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QWidget, QApplication, QCheckBox
from PyQt5.QtGui import QPixmap, QMovie
import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import qdarkstyle
import darkdetect
from HandTracking import HandDetector
import cv2
import pyautogui
import numpy as np
import time
import math
import os


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, inp1, inp2, inp3, inp4, inp5):  # parameters are TRUE/FALSE values from gesture window
        super().__init__()
        self._run_flag = True
        self.leftclick = inp1
        self.scrollup = inp2
        self.scrolldown = inp3
        self.rightclick = inp4
        self.drag = inp5

    def run(self):

        wCam, hCam = 640, 480  # width and height of camera
        frameR = 100  # reduces the part of the viewfinder needed to move the mouse
        smoothenvalue = 7  # smoothens values
        plocX, plocY = 0, 0  # previous x and y coordinates of the mouse
        clocX, clocY = 0, 0  # current x and y coordinates of the mouse
        pTime = 0  # time used for fps calculation
        beforeClicktime = 0  # records time once thumb and index come together
        afterClicktime = 0  # records time once thumb and index finger come apart
        fistactive = 0  # 0/1 value that ensures the program does not repeatedly detect a fist
        clickactive = 1  # 0/1 value that ensures left click is only activated once when click action is performed
        rightclickactive = 1  # 0/1 value that ensures right click is only activated once when right click action is
        # performed

        selectedleftclick = self.leftclick
        selectedscrollup = self.scrollup
        selectedscrolldown = self.scrolldown
        selectedrightclick = self.rightclick
        selecteddrag = self.drag

        detector = HandDetector(detectionCon=0.8)  # Instantiates HandDetector class with a 80% hand confidence

        wScr, hScr = pyautogui.size()  # width and height of the operating systems screen

        # capture from web cam
        cap = cv2.VideoCapture(0)  # stores webcam view in variable
        assert cap.isOpened()
        cap.set(3, wCam)  # sets width of camera view
        cap.set(4, hCam)  # sets height of camera view

        while self._run_flag:
            # Get image frame
            success, img = cap.read()
            if not success:
                break

            # Find the hand and its landmarks
            # hands, img = detector.findHands(img)  # with draw landmark connections
            hands = detector.findHands(img, draw=False)  # without draw landmark connections

            if hands:  # will either have a length of 1 or 2 depending on the number of hands detected

                if len(hands) == 1:
                    # Hand 1

                    hand1 = hands[0]
                    lmList1 = hand1["lmList"]  # List of 21 Landmark points
                    centrePoint1 = hand1['centre']  # centre of the hand cx,cy
                    handType1 = hand1["type"]  # Handtype Left or Right

                    # Defining variables
                    webcamhand1x0 = centrePoint1[0]
                    webcamhand1y0 = centrePoint1[1]
                    webcamhand1x1 = lmList1[8][0]  # landmark 8: tip of index finger
                    webcamhand1y1 = lmList1[8][1]
                    webcamhand1x2 = lmList1[4][0]  # landmark 4: tip of thumb
                    webcamhand1y2 = lmList1[4][1]
                    webcamhand1x3 = lmList1[16][0]  # landmark 16: tip of ring finger
                    webcamhand1y3 = lmList1[16][1]
                    webcamhand1x4 = lmList1[20][0]  # landmark 20: tip of pinky finger
                    webcamhand1y4 = lmList1[20][1]

                    # Drawing on image the box of movement

                    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 0), 2)
                    screenx1 = np.interp(webcamhand1x0, (frameR, wCam - frameR), (0, wScr))  # interpolates screen
                    # coordinates to smaller movement box coordinates
                    screeny1 = np.interp(webcamhand1y0, (frameR, hCam - frameR), (0, hScr))

                    # smoothen values of mouse movement
                    clocX = plocX + (screenx1 - plocX) / smoothenvalue  # reduces shake in mouse movement (tiny hand
                    # vibrations)
                    clocY = plocY + (screeny1 - plocY) / smoothenvalue

                    # moving mouse, if coordinates are outside values then places mouse in the middle of the screen
                    try:
                        pyautogui.moveTo(wScr - clocX, clocY, _pause=False)
                        plocX, plocY = clocX, clocY
                    except ValueError or pyautogui.FailSafeException:
                        pyautogui.moveTo(wScr / 2, hScr / 2, _pause=False)

                    # calculating distance between points for left click + scroll up and down using findDistance method
                    lengththumbindex = detector.findDistance(webcamhand1x1, webcamhand1y1, webcamhand1x2, webcamhand1y2,
                                                             img)
                    lengthringindex = detector.findDistance(webcamhand1x3, webcamhand1y3, webcamhand1x2, webcamhand1y2,
                                                            img)
                    lengthpinkythumb = detector.findDistance(webcamhand1x2, webcamhand1y2, webcamhand1x4, webcamhand1y4,
                                                             img)

                    # if values are below a certain threshold then timer starts (this counts how long the fingers
                    # have stayed together)
                    if lengththumbindex[0] < 40:
                        beforeClicktime = time.time()  # time recorded when thumb and index are close
                    elif lengththumbindex[0] > 40:
                        afterClicktime = time.time()  # time recorded when thumb and index have separated
                    elapsedTime = afterClicktime - beforeClicktime  # find elapsed time
                    elapsedTime = math.trunc(elapsedTime)

                    if lengththumbindex[0] > 40 and clickactive == 1:  # sets variable to 0 when fingers are far away
                        clickactive = 0
                    if lengthpinkythumb[0] > 40 and rightclickactive == 1:  # sets variable to 0 when fingers are far
                        # away
                        rightclickactive = 0

                    if elapsedTime < 0 and lengththumbindex[
                        0] < 40 and selectedscrollup == True:  # if elpased time is zero then click if elapsed time
                        # is negative then scroll
                        pyautogui.scroll(1, _pause=False)
                    elif elapsedTime == 0 and lengththumbindex[0] < 40 and selectedleftclick == True:
                        if clickactive == 0:  # will only click once since clickactive is set to 1, only reset to 0
                            # once fingers are separated
                            pyautogui.click(_pause=False)
                        clickactive = 1
                    elif lengthringindex[0] < 40 and selectedscrolldown == True:
                        pyautogui.scroll(-1, _pause=False)

                    elif lengthpinkythumb[0] < 40 and selectedrightclick == True:
                        if rightclickactive == 0:  # will only click once since rightclickactive is set to 1,
                            # only reset to 0 once fingers are separated
                            pyautogui.rightClick(_pause=False)
                        rightclickactive = 1

                if len(hands) == 2:

                    # Hand 1
                    hand1 = hands[0]
                    lmList1 = hand1["lmList"]  # List of 21 Landmark points
                    centrePoint1 = hand1['centre']  # centre of the hand cx,cy
                    handType1 = hand1["type"]  # Handtype Left or Right

                    # Hand 2
                    hand2 = hands[1]
                    lmList2 = hand2["lmList"]  # List of 21 Landmark points
                    centrePoint2 = hand2['centre']  # centre of the hand cx,cy
                    handType2 = hand2["type"]  # Hand Type "Left" or "Right"

                    webcamhand2x0 = centrePoint2[0]
                    webcamhand2y0 = centrePoint2[1]
                    webcamhand2x1 = lmList2[8][0]  # landmark 8: tip of index finger
                    webcamhand2y1 = lmList2[8][1]
                    webcamhand2x2 = lmList2[4][0]  # landmark 4: tip of thumb
                    webcamhand2y2 = lmList2[4][1]
                    webcamhand2x3 = lmList2[16][0]  # landmark 16: tip of ring finger
                    webcamhand2y3 = lmList2[16][1]
                    webcamhand2x4 = lmList2[20][0]  # landmark 20: tip of pinky finger
                    webcamhand2y4 = lmList2[20][1]

                    YtipIndex = lmList1[8][1]  # for hand 1

                    YmidIndex = lmList1[5][1]  # landmark 5: middle of index finger

                    Ybottom = lmList1[0][1]  # landmark 0: bottom of hand

                    # Drawing on image the box of movement

                    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 0), 2)
                    screenx1 = np.interp(webcamhand2x0, (frameR, wCam - frameR), (0, wScr))
                    screeny1 = np.interp(webcamhand2y0, (frameR, hCam - frameR), (0, hScr))

                    # "smoothenvalue" smoothens the values of mouse movement
                    clocX = plocX + (screenx1 - plocX) / smoothenvalue
                    clocY = plocY + (screeny1 - plocY) / smoothenvalue

                    if (YtipIndex < Ybottom) and (YtipIndex > YmidIndex):  # fist is true if the tip of the index is
                        # higher than bottom of hand and the tip of the index is lower than the middle (higher
                        # coordinates means lower on the screen)
                        fist = True

                    else:
                        fist = False

                    if fist == False:
                        try:
                            fistactive = 0
                            pyautogui.mouseUp(button='left', _pause=False)  # lifts mouse up and moves mouse
                            pyautogui.moveTo(wScr - clocX, clocY, _pause=False)
                            plocX, plocY = clocX, clocY
                        except ValueError or pyautogui.FailSafeException:
                            pyautogui.moveTo(wScr / 2, hScr / 2, _pause=False)
                    elif fist == True and selecteddrag == True:
                        try:
                            if fistactive == 0:  # fistactive set to 1 so cannot be triggered until a fist is opened
                                # again
                                fistactive = 1
                                pyautogui.mouseDown(button='left', _pause=False)  # puts mouse down for text selection
                            else:
                                pyautogui.moveTo(wScr - clocX, clocY, _pause=False)
                                plocX, plocY = clocX, clocY
                        except ValueError or pyautogui.FailSafeException:
                            pyautogui.moveTo(wScr / 2, hScr / 2, _pause=False)
                    elif fist == False and selecteddrag == False:  # fufilling all conditions
                        try:
                            pyautogui.moveTo(wScr - clocX, clocY, _pause=False)
                            plocX, plocY = clocX, clocY
                        except ValueError or pyautogui.FailSafeException:
                            pyautogui.moveTo(wScr / 2, hScr / 2, _pause=False)

                    # calculating distance between points for left click + scroll up and down
                    lengththumbindex = detector.findDistance(webcamhand2x1, webcamhand2y1, webcamhand2x2, webcamhand2y2,
                                                             img)
                    lengthringindex = detector.findDistance(webcamhand2x3, webcamhand2y3, webcamhand2x2, webcamhand2y2,
                                                            img)

                    lengthpinkythumb = detector.findDistance(webcamhand2x2, webcamhand2y2, webcamhand2x4, webcamhand2y4,
                                                             img)

                    # all code below is same as for 1 hand
                    # if values are below a certain threshold then timer starts
                    # (this counts how long the fingers have stayed together)
                    if lengththumbindex[0] < 40:
                        beforeClicktime = time.time()
                    elif lengththumbindex[0] > 40:
                        afterClicktime = time.time()
                    elapsedTime = afterClicktime - beforeClicktime
                    elapsedTime = math.trunc(elapsedTime)

                    if lengththumbindex[0] > 40 and clickactive == 1:
                        clickactive = 0
                    if lengthpinkythumb[0] > 40 and rightclickactive == 1:
                        rightclickactive = 0

                    if elapsedTime < 0 and lengththumbindex[
                        0] < 40 and selectedscrollup == True:
                        pyautogui.scroll(1, _pause=False)

                    elif elapsedTime == 0 and lengththumbindex[0] < 40 and selectedleftclick == True:
                        if clickactive == 0:
                            pyautogui.click(_pause=False)
                        clickactive = 1
                    elif lengthringindex[0] < 40 and selectedscrolldown == True:
                        pyautogui.scroll(-1, _pause=False)
                    elif lengthpinkythumb[0] < 40 and selectedrightclick == True:
                        if rightclickactive == 0:
                            pyautogui.rightClick(_pause=False)
                        rightclickactive = 1

            # fps counter
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,

                        (255, 0, 0), 3)

            if success:
                self.change_pixmap_signal.emit(img)
        # shut down capture system
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class MainWindow(QWidget):
    def __init__(self, window_2):
        super(MainWindow, self).__init__()
        self.window_2 = window_2
        loadUi("mainwindow.ui", self)  # loads mainwindow ui file
        self.setWindowTitle("Hand Tracker")
        self.display_width = 640  # sets camera view
        self.display_height = 480
        self.firsttime = 1
        if darkdetect.isDark() == True:  # checks for dark mode
            self.setStyleSheet(
                qdarkstyle._load_stylesheet(qt_api='pyqt5', palette=None))  # applies dark mode stylesheet
        # start the thread
        self.gesture.clicked.connect(self.gotowindow2)  # executes gotowindow2 method when button pressed
        self.startbutton.clicked.connect(self.startvideo)  # executes startvideo method when button pressed
        # self.darkmodecheck.clicked.connect(self.darkmode)

    def darkmode(self):
        print(darkdetect.isDark())

    def gotowindow2(self):
        widget.setCurrentIndex(widget.currentIndex() + 1)  # changes index by 1 to change page

    def startvideo(self):
        # Change label color to light blue
        self.startbutton.clicked.disconnect(self.startvideo)  # disconnect button from startvideo and link it to
        # stopvideo, button will execute stopvideo on next press

        self.image_label.resize(640, 480)  # resizes image label
        # self.clearlabel.lower()
        # self.clearlabel.lower()
        # Change button to stop
        self.startbutton.setText('Stop Video')
        checkbox = self.window_2.returnvalues()  # gets checkbox values from second window
        # passes through to main program
        self.thread = VideoThread(checkbox[0], checkbox[1], checkbox[2], checkbox[3], checkbox[4])
        self.thread.change_pixmap_signal.connect(self.update_image)  # update image
        # start the thread
        self.thread.start()
        if self.firsttime == 0:  # waits 4 seconds before displaying image if not first time
            time.sleep(4)  # allows for camera to launch
            self.firsttime = 0

        self.startbutton.clicked.connect(self.thread.stop)  # Stop the video if button clicked
        self.startbutton.clicked.connect(self.stopvideo)
        self.firsttime = 0

    def stopvideo(self):
        self.thread.change_pixmap_signal.disconnect()
        self.image_label.resize(0, 0)  # resizes label to 0 to remove camera view
        # self.clearlabel.raise_()
        self.startbutton.setText('Start Video')
        self.startbutton.clicked.disconnect(self.stopvideo)
        self.startbutton.clicked.disconnect(self.thread.stop)
        self.startbutton.clicked.connect(self.startvideo)  # when start button clicked again it triggers startvideo

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, img):  # continual updating of the image (as videos are lots of images)
        qt_img = self.convert_cv_qt(img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, img):  # converts opencv image to qpixmap
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class Button(QCheckBox):
    entered = pyqtSignal()
    left = pyqtSignal()

    def enterEvent(self, event):
        super().enterEvent(event)
        self.entered.emit()  # emit signal when entered

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.left.emit()  # emit signal when left


class Window2(QWidget):
    def __init__(self):
        super(Window2, self).__init__()
        loadUi("window2.ui", self)  # loads window2 ui file
        self.display_width = 800
        self.display_height = 640
        self.backbutton.clicked.connect(self.returnvalues)
        self.backbutton.clicked.connect(self.gotowindow1)  # back

        if darkdetect.isDark() == True:
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        self.leftclickbutton = Button(self)  # defines checkboxes for each function
        self.scrollupbutton = Button(self)
        self.scrolldownbutton = Button(self)
        self.rightclickbutton = Button(self)
        self.dragbutton = Button(self)
        # leftclickbutton.setGeometry(QtCore.QRect(130, 80, 381, 51))

        ###########CODE FOR LEFT CLICK BUTTON#############

        self.leftclickbutton.setGeometry(QtCore.QRect(60, 40, 251, 51))
        if darkdetect.isDark() == False:  # if light mode then apply one stylesheet
            self.leftclickbutton.setStyleSheet("QCheckBox{\n"
                                               "padding: 10px;\n"
                                               "}\n"
                                               "\n"
                                               "\n"
                                               "QCheckBox::hover {\n"
                                               "    border: 1px #d0d0d0 ;\n"
                                               "    background-color: #D9D9D9 ;\n"
                                               "    border-radius: 10px;\n"
                                               "    padding: 10px;\n"
                                               "\n"
                                               "\n"
                                               "}\n"
                                               "")
        else:
            self.leftclickbutton.setStyleSheet("QCheckBox{\n"  # if dark mode then apply one stylesheet
                                               "padding: 10px;\n"
                                               "background-color: #323232 ; \n"
                                               "}\n"
                                               "\n"
                                               "\n"
                                               "QCheckBox::hover {\n"
                                               "    border: 1px #d0d0d0 ;\n"
                                               "    background-color: #454445 ;\n"
                                               "    border-radius: 10px;\n"
                                               "    padding: 10px;\n"
                                               "\n"
                                               "\n"
                                               "}\n"
                                               "")

        self.leftclickbutton.setChecked(True)  # set to enabled by default
        self.leftclickbutton.setTristate(False)
        # increase clickable area for the button
        self.leftclickbutton.setText("Left Click                                                             ")
        self.leftclickbutton.setObjectName("leftclickbutton")

        ##########CODE FOR SCROLL UP BUTTON##############
        self.scrollupbutton.setGeometry(QtCore.QRect(60, 90, 251, 51))
        if darkdetect.isDark() == False:
            self.scrollupbutton.setStyleSheet("QCheckBox{\n"
                                              "padding: 10px;\n"
                                              "}\n"
                                              "\n"
                                              "\n"
                                              "QCheckBox::hover {\n"
                                              "    border: 1px #d0d0d0 ;\n"
                                              "    background-color: #D9D9D9 ;\n"
                                              "    border-radius: 10px;\n"
                                              "    padding: 10px;\n"
                                              "\n"
                                              "\n"
                                              "}\n"
                                              "")
        else:
            self.scrollupbutton.setStyleSheet("QCheckBox{\n"
                                              "padding: 10px;\n"
                                              "background-color: #323232 ; \n"
                                              "}\n"
                                              "\n"
                                              "\n"
                                              "QCheckBox::hover {\n"
                                              "    border: 1px #d0d0d0 ;\n"
                                              "    background-color: #454445 ;\n"
                                              "    border-radius: 10px;\n"
                                              "    padding: 10px;\n"
                                              "\n"
                                              "\n"
                                              "}\n"
                                              "")
        self.scrollupbutton.setChecked(True)
        self.scrollupbutton.setTristate(False)
        self.scrollupbutton.setText("Scroll Up                                                         ")
        self.scrollupbutton.setObjectName("scrollupbutton")

        ##########Code for scroll down button###########
        self.scrolldownbutton.setGeometry(QtCore.QRect(60, 140, 251, 51))

        if darkdetect.isDark() == False:
            self.scrolldownbutton.setStyleSheet("QCheckBox{\n"
                                                "padding: 10px;\n"
                                                "}\n"
                                                "\n"
                                                "\n"
                                                "QCheckBox::hover {\n"
                                                "    border: 1px #d0d0d0 ;\n"
                                                "    background-color: #D9D9D9 ;\n"
                                                "    border-radius: 10px;\n"
                                                "    padding: 10px;\n"
                                                "\n"
                                                "\n"
                                                "}\n"
                                                "")
        else:
            self.scrolldownbutton.setStyleSheet("QCheckBox{\n"
                                                "padding: 10px;\n"
                                                "background-color: #323232 ; \n"
                                                "}\n"
                                                "\n"
                                                "\n"
                                                "QCheckBox::hover {\n"
                                                "    border: 1px #d0d0d0 ;\n"
                                                "    background-color: #454445 ;\n"
                                                "    border-radius: 10px;\n"
                                                "    padding: 10px;\n"
                                                "\n"
                                                "\n"
                                                "}\n"
                                                "")
        self.scrolldownbutton.setChecked(True)
        self.scrolldownbutton.setTristate(False)
        self.scrolldownbutton.setText("Scroll Down                                   ")
        self.scrolldownbutton.setObjectName("scrolldownbutton")

        ##########Code for right click button###########
        self.rightclickbutton.setGeometry(QtCore.QRect(60, 190, 251, 51))

        if darkdetect.isDark() == False:
            self.rightclickbutton.setStyleSheet("QCheckBox{\n"
                                                "padding: 10px;\n"
                                                "}\n"
                                                "\n"
                                                "\n"
                                                "QCheckBox::hover {\n"
                                                "    border: 1px #d0d0d0 ;\n"
                                                "    background-color: #D9D9D9 ;\n"
                                                "    border-radius: 10px;\n"
                                                "    padding: 10px;\n"
                                                "\n"
                                                "\n"
                                                "}\n"
                                                "")
        else:
            self.rightclickbutton.setStyleSheet("QCheckBox{\n"
                                                "padding: 10px;\n"
                                                "background-color: #323232 ; \n"
                                                "}\n"
                                                "\n"
                                                "\n"
                                                "QCheckBox::hover {\n"
                                                "    border: 1px #d0d0d0 ;\n"
                                                "    background-color: #454445 ;\n"
                                                "    border-radius: 10px;\n"
                                                "    padding: 10px;\n"
                                                "\n"
                                                "\n"
                                                "}\n"
                                                "")
        self.rightclickbutton.setChecked(True)
        self.rightclickbutton.setTristate(False)
        self.rightclickbutton.setText("Right Click                                   ")
        self.rightclickbutton.setObjectName("rightclickbutton")

        ##########Code for drag button###########
        self.dragbutton.setGeometry(QtCore.QRect(60, 240, 251, 51))

        if darkdetect.isDark() == False:
            self.dragbutton.setStyleSheet("QCheckBox{\n"
                                          "padding: 10px;\n"
                                          "}\n"
                                          "\n"
                                          "\n"
                                          "QCheckBox::hover {\n"
                                          "    border: 1px #d0d0d0 ;\n"
                                          "    background-color: #D9D9D9 ;\n"
                                          "    border-radius: 10px;\n"
                                          "    padding: 10px;\n"
                                          "\n"
                                          "\n"
                                          "}\n"
                                          "")
        else:
            self.dragbutton.setStyleSheet("QCheckBox{\n"
                                          "padding: 10px;\n"
                                          "background-color: #323232 ; \n"
                                          "}\n"
                                          "\n"
                                          "\n"
                                          "QCheckBox::hover {\n"
                                          "    border: 1px #d0d0d0 ;\n"
                                          "    background-color: #454445 ;\n"
                                          "    border-radius: 10px;\n"
                                          "    padding: 10px;\n"
                                          "\n"
                                          "\n"
                                          "}\n"
                                          "")
        self.dragbutton.setChecked(True)
        self.dragbutton.setTristate(False)
        self.dragbutton.setText("Drag                                   ")
        self.dragbutton.setObjectName("dragbutton")

        self.leftclickbutton.entered.connect(
            self.leftclickbuttonclicked)  # when hovered over execute leftclickbuttonclicked
        # leftclickbutton.left.connect(self.stopvideo)
        self.scrollupbutton.entered.connect(self.scrollupbuttonclicked)
        self.scrolldownbutton.entered.connect(self.scrolldownbuttonclicked)
        self.leftclickbutton.left.connect(self.stopvideo)  # when left execute stopvideo
        self.scrollupbutton.left.connect(self.stopvideo)
        self.scrolldownbutton.left.connect(self.stopvideo)
        self.rightclickbutton.entered.connect(self.rightclickbuttonclicked)
        self.rightclickbutton.left.connect(self.stopvideo)
        self.dragbutton.entered.connect(self.dragbuttonclicked)
        self.dragbutton.left.connect(self.stopvideo)

    def leftclickbuttonclicked(self):
        self.movie = QMovie("leftclick.gif")  # play leftclickgif when executed
        self.label.setMovie(self.movie)  # set a label to the gif
        self.movie.start()  # start movie

    def scrollupbuttonclicked(self):
        self.movie = QMovie("scrollup.gif")
        self.label.setMovie(self.movie)
        self.movie.start()

    def scrolldownbuttonclicked(self):
        self.movie = QMovie("scrolldown.gif")
        self.label.setMovie(self.movie)
        self.movie.start()

    def rightclickbuttonclicked(self):
        self.movie = QMovie("rightclick.gif")
        self.label.setMovie(self.movie)
        self.movie.start()

    def dragbuttonclicked(self):
        self.movie = QMovie("drag.gif")
        self.label.setMovie(self.movie)
        self.movie.start()

    def stopvideo(self):
        self.movie.stop()

    def gotowindow1(self):
        widget.setCurrentIndex(widget.currentIndex() - 1)

    def returnvalues(self):
        left = self.leftclickbutton.isChecked()  # return true false state of button
        scrollup = self.scrollupbutton.isChecked()
        scrolldown = self.scrolldownbutton.isChecked()
        rightclick = self.rightclickbutton.isChecked()
        drag = self.dragbutton.isChecked()
        checkboxvalues = [left, scrollup, scrolldown, rightclick, drag]
        return checkboxvalues


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()
    b = Window2()
    a = MainWindow(b)  # main window inherits window 2
    widget.setFixedSize(1000, 600)
    widget.addWidget(a)
    widget.addWidget(b)
    widget.show()
    sys.exit(app.exec_())
