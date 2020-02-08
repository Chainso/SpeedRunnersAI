import torch
import torch.nn as nn

from pykeyboard import PyKeyboardEvent
from threading import Thread
from multiprocessing import Process

from speedrunners import SpeedRunnersEnv

def _loop(listener):
    action = torch.FloatTensor([0.5] * 7)
    action = torch.distributions.Bernoulli(action)
    while(listener.running):
        while(listener.playing):
        
            state = listener.env.reset()
            while(listener.playing):
                state = listener.env.state
                act = action.sample().numpy()
                next_state, reward, terminal = listener.env.step(act)

class Listener(PyKeyboardEvent):
    def __init__(self):
        PyKeyboardEvent.__init__(self)

        game_screen = (1920, 1080)
        res_size = (128, 128, 1)
    
        self.env = SpeedRunnersEnv(120, game_screen, res_size)

        self.running = True
        self.playing = False

        #thread = Thread(target = self._loop)

        #thread.start()


    def tap(self, keycode, character, press):
        if(press):
            if(character == "n"):
                self.playing = not self.playing

                if(not self.playing):
                    self.env.pause()
                else:
                    self.env.start()

                print("Playing:", self.playing)

                if(not self.running):
                    self.running = True

                    #thread = Thread(target = self._loop)
                    #thread.start()

            elif(character == "l"):
                print("Stopping")
                self.playing = False
                self.running = False
                self.env.stop()
                self.stop()
                print("Stopped?")

if(__name__ == "__main__"):
    listener = Listener()

    main_loop = Thread(target = _loop, args = (listener,))
    main_loop.start()

    listener.run()

    listener.join()
    main_loop.join()
    print("End of program")
    """
    game_screen = (1920, 1080)
    res_size = (128, 128, 1)

    env = SpeedRunnersEnv(120, game_screen, res_size)
    env.start()
    env.stop()

    action = torch.FloatTensor([0.5] * 7)
    action = torch.distributions.Bernoulli(action)

    state = env.reset()
    while(True):
        act = action.sample().numpy()
        next_state, reward, terminal = ennnv.step(act)
        state = next_state
    """