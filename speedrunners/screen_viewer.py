import numpy as np
import cv2
import time

from mss import mss
from PIL import Image
from threading import Thread, Lock

# Asynchronously captures screens of a window. Provides functions for accessing
# the captured screen.
class ScreenViewer:
 
    def __init__(self, window_size, res_shape, update_hook = None):
        self.res_width, self.res_height, self.res_depth = res_shape
        self.width, self.height = window_size
        self.mut = Lock()
        self.hwnd = None
        self.its = None         #Time stamp of last image
        self.i0 = None          #i0 is the latest image; 
        self.i1 = None          #i1 is used as a temporary variable
        self.cl = False         #Continue looping flag
        self.screen = mss()
        self.monitor = {'top': 0, 'left': 0, 'width': self.width,
                        'height': self.height}

        # If the latest screen was polled
        self.screen_polled = False

        self.update_hook = update_hook

    def set_polled(self):
        """
        Sets the current screen to have been polled.
        """
        return self.screen_polled

    #Get's the latest image of the window
    def GetScreen(self):
        while self.i0 is None:      #Screen hasn't been captured yet
            pass
        self.mut.acquire()
        s = self.i0
        self.set_polled()
        self.mut.release()
        return s

    def GetNewScreen(self):
        while(self.screen_polled):
            pass

        return self.GetScreen()

    #Get's the latest image of the window along with timestamp
    def GetScreenWithTime(self):
        while self.i0 is None:      #Screen hasn't been captured yet
            pass
        self.mut.acquire()
        s = self.i0
        t = self.its
        self.mut.release()
        return s, t
         
    # Gets the screen of the window referenced by self.hwnd
    def GetScreenImg(self):
        sct_img = self.screen.grab(self.monitor)
        im = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        im = np.array(im)

        if(self.res_depth == 0 or self.res_depth == 1):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (self.res_width, self.res_height))

            if(self.res_depth == 1):
                im = np.expand_dims(im, 3)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (self.res_width, self.res_height))

        return im

    #Begins recording images of the screen
    def Start(self):
        #if self.hwnd is None:
        #    return False
        self.cl = True
        time.sleep(1.5)

        thrd = Thread(target = self.ScreenUpdateT)
        thrd.start()

        return True
         
    #Stop the async thread that is capturing images
    def Stop(self):
        self.cl = False
        self.i0 = False

    #Thread used to capture images of screen
    def ScreenUpdateT(self):
        """
        Updates the screen with the new image captures. Will call the update
        hook with the new screen as the parameter if a hook is provided.
        """
        #Keep updating screen until terminating
        while self.cl:
            self.i1 = self.GetScreenImg()
            self.mut.acquire()
            self.i0 = self.i1               #Update the latest image in a thread safe way
            self.screen_polled = False
            self.update_hook(self.i0)
            self.its = time.time()
            self.mut.release()
