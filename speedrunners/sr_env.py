import numpy as np

from speedrunners.screen_viewer import ScreenViewer
from speedrunners.memory_reader import MemoryReader
from speedrunners.actor import Actor

class SpeedRunnersEnv():
    # The addresses of all the values to get
    addresses = {"x_speed" : 0x29CBF23C,
                 "y_speed" : 0x29CBF240,
                 "current_time" : 0x29C9584C,
                 "obstacles_hit" : 0x739D1334}

    # The offsets of all the values to get
    offsets = {"obstacles_hit" : [0x0, 0x0, 0x1F6C, 0x8, 0x4]}

    def __init__(self, max_time, window_size, res_shape):
        self.actor = Actor()
        self.sv = ScreenViewer(window_size, res_shape, self._screen_update_hook)
        self.memory = MemoryReader("speedrunners.exe")
        self.max_time = max_time
        self.num_actions = self.actor.num_actions()

        # The environment properties
        self._state = None
        self._reward = 0
        self._terminal = False
        self._reached_goal = False

        self._last_time = 0
        self.num_obstacles_hit = self.memory.get_obstacles_hit()

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
        self.start()

        self.actor.release_keys()
        self._episode_finished(True)
        self.sv.set_polled()
        self.sv.GetNewScreen()

        return self.state

    def start(self):
        """
        Starts the screen viewer.
        """
        self.sv.Start()

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
        self.memory.close_handle()

    def step(self, action):
        """
        Takes a step into the environment with the action given.
        """
        self.sv.set_polled()
        self.actor.act(action)
        self._state = self.sv.GetNewScreen()

        return self.state, self.reward, self.terminal

    def _get_reward(self):
        """
        Returns the reward for the current state of the environment.
        """
        reward = -1 / (self._get_speed() + 1)

        obst_dif = self._get_obstacles_hit() - self.num_obstacles_hit

        reward -= 0.05 * obst_dif
        self.num_obstacles_hit += obst_dif

        if(self._reached_goal):
            reward += 10
        elif(self._terminal):
            reward = 0

        self._reward = reward

    def _get_x_speed(self):
        return self.memory.get_address(self.addresses["x_speed"], c_float)

    def _get_y_speed(self):
        return self.memory.get_address(self.addresses["y_speed"], c_float)

    def _get_speed(self):
        return np.sqrt(np.square(self.get_x_speed()) + 0.75 * np.square(self.get_y_speed()))

    def _get_current_time(self):
        return self.memory.get_address(self.addresses["current_time"], c_float)

    def _get_obstacles_hit(self):
        return self.memory.get_address(self.addresses["obstacles_hit"], ctypes.c_byte)

    def _episode_finished(self, reset = False):
        if((self.max_time is not None
            and self._get_current_time() > self.max_time)
           or reset):
            self._last_time = 0
            self._terminal = True
            self._reached_goal = False
        elif(self._get_current_time() < self._last_time):
            self._last_time = 0
            self._terminal = True
            self._reached_goal = True
        else:
            self._last_time = self._get_current_time()
            self._terminal = False
            self._reached_goal = False

        if(self._terminal and not self._reached_goal and not reset):
            self.sv.stop()

    def state_space(self):
        return self.res_shape

    def action_space(self):
        return self.num_actions
