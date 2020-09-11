import numpy as np
import d3dshot

from time import time
from pymem import Pymem
from torch.nn.functional import interpolate

#from speedrunnersai.speedrunners.screen_viewer import ScreenViewer
from speedrunnersai.speedrunners.memory_reader import MemoryReader
from speedrunnersai.speedrunners.actor import Actor
from speedrunnersai.speedrunners.structures import Address, Player, Match

class SpeedRunnersEnv():
    # The addresses of all the values to get
    addresses = {"x_speed" : 0x29CBF23C,
                 "y_speed" : 0x29CBF240,
                 "current_time" : 0x29C9584C,
                 "obstacles_hit" : 0x739D1334}

    # The offsets of all the values to get
    offsets = {"obstacles_hit" : [0x0, 0x0, 0x1F6C, 0x8, 0x4]}

    def __init__(self, max_time, window_size, res_shape, read_mem = False):
        self.res_shape = res_shape
        self.actor = Actor()

        # Create the d3dshot instance
        d3dshot = d3dshot.create(capture_output="pytorch_float_gpu")
        formula = torch.FloatTensor([0.2125, 0.7154, 0.0721]).to("cuda")
        grayscale_transform = lambda frame: do_mul(frame, formula).unsqueeze(1)
        interpolate_transform = lambda frame: interpolate(
            frame, size=(128, 128), mode="bilinear"
        )
        
        transforms = [grayscale_transform]#, interpolate_transform]
        capture_time = 10
        frame_handler = FrameHandler(d3d, transforms=transforms)

        #self.sv = ScreenViewer(window_size, res_shape, self._screen_update_hook)
        
        self.max_time = max_time
        self.num_actions = self.actor.num_actions()

        # The environment properties
        self._state = None
        self._reward = 0
        self._terminal = False
        self._reached_goal = False

        # Relevant memory information
        self.memory = Pymem() if read_mem else None
        self.match = Match()
        self.match.players.append(Player())

        self._last_time = 0
        self.num_obstacles_hit = self._get_obstacles_hit()
        self.start_time = 0

    @property
    def state(self):
        """
        The current state of the environment. WIll be None if the environment
        has not been started.
        """
        return self._state

    @property
    def reward(self):
        """
        Returns the reward of the last action taken. If no previous action has
        been taken then 0 will be returned.
        """
        return self._reward

    @property
    def terminal(self):
        """
        Returns true if the current state is a terminal state.
        """
        return self._terminal

    def _screen_update_hook(self, new_state):
        """
        Will update the current state whenever a new screen is captured.
        """
        self._state = new_state
        self._episode_finished()
        self._get_reward()

    def reset(self):
        """
        Resets the game (in practice).
        """
        self.actor.reset()
        self._episode_finished(True)
        self.sv.set_polled()
        self.sv.GetNewScreen()
        self.start_time = time()

        return self.state

    def start(self):
        """
        Starts the screen viewer.
        """
        self.sv.Start()

        if self.memory is not None:
            self.memory.open_process_from_name("speedrunners.exe")

    def pause(self):
        """
        Closes the screen viewer.
        """
        self.sv.Stop()

    def stop(self):
        """
        Closes the environment.
        """
        self.reset()
        self.sv.Stop()
        self.actor.stop()

        if self.memory is not None:
            self.memory.close_handle()

    def step(self, action):
        """
        Takes a step into the environment with the action given.
        """
        self.sv.set_polled()
        self.actor.act(action)
        self._state = self.sv.GetNewScreen()

        # Update structures with new memory
        self._update_memory()

        # Dont forget to calculate rewards
        self._episode_finished(False)

        return self.state, self.reward, self.terminal

    def _update_memory(self):
        """
        Updates match with the new values of the process.
        """
        if self.memory is not None:
            self.match.update(self.memory, 0)

    def _get_reward(self):
        """
        Returns the reward for the current state of the environment.
        """
        reward = 0

        reward = self.match.players[0].speed() / 700

        obst_dif = self._get_obstacles_hit() - self.num_obstacles_hit

        obst_reward = -0.05 * obst_dif
        self.num_obstacles_hit += obst_dif

        if(self._reached_goal):
            reward += 10
        elif(self._terminal):
            reward = obst_reward

        self._reward = reward


    def _get_obstacles_hit(self):
        """
        return (self.memory.get_address(self.addresses["obstacles_hit"],
                                        ctypes.c_byte)
                if self.memory is not None else 0)
        """
        return 0


    def _episode_finished(self, reset = False):
        if((self.max_time is not None
            and self.match.current_time > self.max_time)
            or (self.max_time is not None
                and (time() - self.start_time) > self.max_time)
            or reset):
            self._last_time = 0
            self._terminal = True
            self._reached_goal = False
        elif(self.match.current_time < self._last_time):
            self._last_time = 0
            self._terminal = True
            self._reached_goal = True
        else:
            self._last_time = self.match.current_time
            self._terminal = False
            self._reached_goal = False

    def state_space(self):
        """
        Returns the shape of the captured images.
        """
        return self.res_shape

    def action_space(self):
        """
        Returns the number of actions the agent can take.
        """
        return self.num_actions
