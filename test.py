from pykeyboard import PyKeyboard, PyKeyboardEvent
from time import sleep
from threading import Thread

class A(PyKeyboardEvent):
    def __init__(self):
        PyKeyboardEvent.__init__(self)

        self.keyboard = PyKeyboard()
        self.continue_running = False
        self.loop = True

        t = Thread(target = self._loop)
        t.start()

    def _loop(self):
        while(self.loop):
            while(self.continue_running):
                self.keyboard.tap_key("aaal")
                sleep(1)

    def tap(self, keycode, character, press):
        print(character)
        if(press and character == "l"):
            print("Yea")
            self.continue_running = not self.continue_running

if(__name__ == "__main__"):
    a = A()
    a.run()