from binding.python.porcupine import Porcupine
import pyaudio
import struct
import time
import ctypes
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
import time
import pyautogui
from matplotlib import pyplot as plt
import win32api, win32con
global bg
global x_move, y_move
bg = None

SendInput = ctypes.windll.user32.SendInput

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
RELOAD = 0x13
LEFT_CLICK = 0x17
RIGHT_CLICK = 0x18
THROW_GRENADE = 0x15

G = 0x22
JUMP_NOW = 0x39
CROUCH_DOWN = 0x2E
CHANGE_WEAPON = 0x02
# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

class ProcessMain:
    def click(self, x, y):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)


#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

# porcupine_wakeword = Porcupine(library_path='lib\\windows\\amd64\\libpv_porcupine.dll',
#                                model_file_path='lib\\common\\porcupine_params.pv',
#                                keyword_file_paths=['resources\\keyword_files\\windows\\porcupine_windows.ppn'],
#                                sensitivities=[0.5])



if __name__ == "__main__":
    # file1 = open("D:/myfile.txt", "w")
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # initialize num of frames
    num_frames = 0
    # initialise t1 = 0
    t1 = 0
    pyautogui.moveTo(640, 480)
    x_old, y_old = pyautogui.position()
    cx,cy = (0,0)
    k = 0

    porcupine_commands1 = Porcupine(library_path='lib\\windows\\amd64\\libpv_porcupine.dll',
                                    model_file_path='lib\\common\\porcupine_params.pv',
                                    keyword_file_paths=['resources\\keyword_files\\change_weapon_windows.ppn'],
                                    sensitivities=[0.5])

    porcupine_commands2 = Porcupine(library_path='lib\\windows\\amd64\\libpv_porcupine.dll',
                                    model_file_path='lib\\common\\porcupine_params.pv',
                                    keyword_file_paths=['resources\\keyword_files\\jump_now_windows.ppn'],
                                    sensitivities=[0.5])

    porcupine_commands3 = Porcupine(library_path='lib\\windows\\amd64\\libpv_porcupine.dll',
                                    model_file_path='lib\\common\\porcupine_params.pv',
                                    keyword_file_paths=['resources\\keyword_files\\reload_windows.ppn'],
                                    sensitivities=[0.5])

    porcupine_commands4 = Porcupine(library_path='lib\\windows\\amd64\\libpv_porcupine.dll',
                                    model_file_path='lib\\common\\porcupine_params.pv',
                                    keyword_file_paths=['resources\\keyword_files\\throw_grenade_windows.ppn'],
                                    sensitivities=[0.5])

    porcupine_commands5 = Porcupine(library_path='lib\\windows\\amd64\\libpv_porcupine.dll',
                                    model_file_path='lib\\common\\porcupine_params.pv',
                                    keyword_file_paths=['resources\\keyword_files\\crouch_down_windows.ppn'],
                                    sensitivities=[0.5])
    # setup audio
    pa1 = pyaudio.PyAudio()
    audio_stream1 = pa1.open(
        rate=porcupine_commands1.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine_commands1.frame_length)

    # keep looping, until interrupted
    while (True):
        # audio processing
        pcm1 = audio_stream1.read(porcupine_commands1.frame_length)
        pcm1 = struct.unpack_from("h" * porcupine_commands1.frame_length, pcm1)

        result1 = porcupine_commands1.process(pcm1)
        if result1:
            # Change Weapon
            PressKey(CHANGE_WEAPON)
            ReleaseKey(CHANGE_WEAPON)

        result2 = porcupine_commands2.process(pcm1)
        if result2:
            #Jump Now
            PressKey(JUMP_NOW)
            ReleaseKey(JUMP_NOW)

        result3 = porcupine_commands3.process(pcm1)
        if result3:
            # RELOAD
            PressKey(RELOAD)
            ReleaseKey(RELOAD)

        result4 = porcupine_commands4.process(pcm1)
        if result4:
            # THROW GRENADE
            PressKey(THROW_GRENADE)
            ReleaseKey(THROW_GRENADE)

        result5 = porcupine_commands5.process(pcm1)
        if result5:
            # Crouch Down
            PressKey(CROUCH_DOWN)
            ReleaseKey(CROUCH_DOWN)
        # t1 = time.time()
        area_yellow=0
        # get the current frame
        (grabbed, frame) = camera.read()

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # get the ROI
        roi = frame

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 10:
            run_avg(gray, aWeight)
        else:

            # find the absolute difference between background and current frame
            diff = cv2.absdiff(bg.astype("uint8"), gray)

            # threshold the diff image so that we get the foreground
            thresholded = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]
            thresholded_dilate = cv2.dilate(thresholded, kernel=np.ones((5, 5)), iterations=3)
            thresholded_dilate = cv2.erode(thresholded_dilate, kernel=np.ones((3, 3)), iterations=1)
            masked_image = cv2.bitwise_and(roi, roi, mask=thresholded_dilate)
            masked_image_hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

            # plt.imshow(masked_image_hsv, cmap='hsv')
            # plt.show()

            mask_pink = cv2.inRange(masked_image_hsv, (140, 40, 50), (184, 255, 255))
            mask_pink = cv2.erode(mask_pink, kernel = np.ones((3,3)), iterations = 2)

            # cv2.imshow("mask_pink", mask_pink)

            mask_yellow = cv2.inRange(masked_image_hsv, (26, 80, 120), (32, 255, 255))
            mask_yellow = cv2.erode(mask_yellow, kernel=np.ones((3, 3)), iterations=2)
            mask_yellow = cv2.dilate(mask_yellow, kernel=np.ones((3, 3)), iterations=2)

            mask_green = cv2.inRange(masked_image_hsv, (0, 180, 100), (8, 255, 200))
            mask_green = cv2.dilate(mask_green, kernel=np.ones((3, 3)), iterations=2)
            cv2.imshow("green_mask", mask_green)

            cv2.imshow("yellow_mask", mask_yellow)

            # the mask of pink color is now created now we have to draw contour around it to find its center
            try:
                contour_pink, heirarchy_pink = cv2.findContours(mask_pink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                areas_pink = [cv2.contourArea(c) for c in contour_pink]
                max_index_pink = np.argmax(areas_pink)
                cnt_pink = contour_pink[max_index_pink]
                area = cv2.contourArea(cnt_pink)
                M_pink = cv2.moments(cnt_pink)
                cx = int(M_pink['m10'] / M_pink['m00'])
                cy = int(M_pink['m01'] / M_pink['m00'])
                centroid = (cx, cy)


                # print(area)

                x_new, y_new = centroid
                x_new = (x_new / 640) * 1920
                y_new = (y_new / 480) * 1080
                offset_x = x_new - x_old
                offset_y = y_new - y_old
                x_move = int(640 + offset_x)
                y_move = int(480 + offset_y)
                # pyautogui.moveTo(640 + offset_x, 480 + offset_y)

                win32api.SetCursorPos((x_move, y_move))
                # print("hello")
                (x_old, y_old) = centroid
                x_old = (x_old / 640) * 1920
                y_old = (y_old / 480) * 1080

                if area>2000:
                    PressKey(W)
                elif area<1000:
                    PressKey(S)
                else:
                    ReleaseKey(W)
                    ReleaseKey(S)
                try:
                    contour_yellow, heirarchy_yellow = cv2.findContours(mask_yellow, cv2.RETR_TREE,
                                                                        cv2.CHAIN_APPROX_SIMPLE)
                    areas_yellow = [cv2.contourArea(c) for c in contour_yellow]

                    max_index_yellow = np.argmax(areas_yellow)
                    cnt_yellow = contour_yellow[max_index_yellow]

                    area_yellow = cv2.contourArea(cnt_yellow)
                    print("area_yello : ", area_yellow)
                except:
                    pass

                try:
                    contour_green, heirarchy_green = cv2.findContours(mask_green, cv2.RETR_TREE,
                                                                        cv2.CHAIN_APPROX_SIMPLE)
                    areas_green = [cv2.contourArea(c) for c in contour_green]

                    max_index_green = np.argmax(areas_green)
                    cnt_green = contour_green[max_index_green]

                    area_green = cv2.contourArea(cnt_green)
                    if area_green > 300:
                        if z==0:
                            z=1
                            PressKey(G)
                except:
                    ReleaseKey(G)
                    z=0
                    pass
                if (area_yellow > 170):
                    # if k == 0:
                    #     k = 1
                    PressKey(LEFT_CLICK)

                else:
                    # if k == 1:
                    ReleaseKey(LEFT_CLICK)
                        # k = 0



            except:
                pass


        # increment the number of frames
        num_frames += 1
        # display the frame with segmented hand
        cv2.imshow("Video Feed", frame)
        keypress = cv2.waitKey(1) & 0xFF
        t2 = time.time()
        # file1.write(str(t2-t1)+'\n')
        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
# free up memory
# file1.close()
camera.release()
cv2.destroyAllWindows()