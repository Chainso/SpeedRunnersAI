from screen_viewer import ScreenViewer
from time import time
import cv2

class Counter():
    def __init__(self):
        self.count = 0

    def increment(self, screen):
        self.count += 1
        print(self.count, "\n")

if(__name__ == "__main__"):
    counter = Counter()

    window_size = (1440, 900)
    res_shape = (128, 128, 1)
    update_hook = counter.increment
    sv = ScreenViewer(window_size, res_shape, update_hook)

    sv.Start()

    total = time()

    while(counter.count < 100):
        start = time()
        screen = sv.GetNewScreen()
        sv.set_polled()
    
        print(time() - start)

    sv.Stop()
    average =  (time() - total)/counter.count
    print("Average time:", average)
    print("FPS:", 1/average)