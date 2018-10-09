import numpy as np
import win32gui
import win32ui, win32con
from threading import Thread, Lock
import time
import cv2

#Asynchronously captures screens of a window. Provides functions for accessing
#the captured screen.
class ScreenViewer:
 
    def __init__(self, res_width, res_height, res_depth=0):
        self.res_width = res_width
        self.res_height = res_height
        self.res_depth = res_depth
        self.num_img = 0
        self.mut = Lock()
        self.hwnd = None
        self.its = None         #Time stamp of last image 
        self.i0 = None          #i0 is the latest image; 
        self.i1 = None          #i1 is used as a temporary variable
        self.cl = False         #Continue looping flag
        #Left, Top, Right, and bottom of the screen window
        self.l, self.t, self.r, self.b = 0, 0, 0, 0
        #Border on left and top to remove
        self.bl, self.bt, self.br, self.bb = 12, 31, 12, 20
 
    #Gets handle of window to view
    #wname:         Title of window to find
    #Return:        True on success; False on failure
    def GetHWND(self, wname):
        self.hwnd = win32gui.FindWindow(None, wname)
        if self.hwnd == 0:
            self.hwnd = None
            return False
        self.l, self.t, self.r, self.b = win32gui.GetWindowRect(self.hwnd)
        return True
         
    #Get's the latest image of the window
    def GetScreen(self):
        while self.i0 is None:      #Screen hasn't been captured yet
            pass
        self.mut.acquire()
        s = self.i0
        self.mut.release()
        return s
         
    #Get's the latest image of the window along with timestamp
    def GetScreenWithTime(self):
        while self.i0 is None:      #Screen hasn't been captured yet
            pass
        self.mut.acquire()
        s = self.i0
        t = self.its
        self.mut.release()
        return s, t
         
    #Gets the screen of the window referenced by self.hwnd
    def GetScreenImg(self):
        if self.hwnd is None:
            raise Exception("HWND is none. HWND not called or invalid window name provided.")
        self.l, self.t, self.r, self.b = win32gui.GetWindowRect(self.hwnd)
        #Remove border around window (8 pixels on each side)
        #Remove 4 extra pixels from left and right 16 + 8 = 24
        w = self.r
        #Remove border on top and bottom (31 on top 8 on bottom)
        #Remove 12 extra pixels from bottom 39 + 12 = 51
        h = self.b
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        #First 2 tuples are top-left and bottom-right of destination
        #Third tuple is the start position in source
        cDC.BitBlt((0,0), (w, h), dcObj, (self.bl, self.bt), win32con.SRCCOPY)
        bmInfo = dataBitMap.GetInfo()
        im = np.frombuffer(dataBitMap.GetBitmapBits(True), dtype = np.uint8)
        im.shape = (h,w,4)

        if(self.res_depth == 0 or self.res_depth == 1):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (self.res_width, self.res_height))

            if(self.res_depth == 1):
                im = np.expand_dims(im, 3)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (self.res_width, self.res_height))

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        #Bitmap has 4 channels like: BGRA. Discard Alpha and flip order to RGB
        #For 800x600 images:
        #Remove 12 pixels from bottom + border
        #Remove 4 pixels from left and right + border
        #return im.reshape(bmInfo['bmHeight'], bmInfo['bmWidth'], 4)[:, :, -2::-1]
        return im

    #Begins recording images of the screen
    def Start(self):
        #if self.hwnd is None:
        #    return False
        self.cl = True

        # Switch the window
        win32gui.SetForegroundWindow(self.hwnd)
        time.sleep(1.5)

        thrd = Thread(target = self.ScreenUpdateT)
        thrd.start()

        return True
         
    #Stop the async thread that is capturing images
    def Stop(self):
        self.cl = False
         
    #Thread used to capture images of screen
    def ScreenUpdateT(self):
        #Keep updating screen until terminating
        while self.cl:
            self.i1 = self.GetScreenImg()
            self.num_img += 1
            self.mut.acquire()
            self.i0 = self.i1               #Update the latest image in a thread safe way
            self.its = time.time()

            self.mut.release()

if __name__ == "__main__":
    sv = ScreenViewer()
    sv.GetHWND('SpeedRunners')
    sv.Start()
    t = 1
    time.sleep(t)
    sv.Stop()
