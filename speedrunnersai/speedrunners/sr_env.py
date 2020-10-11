import numpy as np
import d3dshot
import win32gui

from time import time
from pymem import Pymem
from typing import Tuple, Optional, Any
from hlrl.core.envs import Env
from hlrl.core.vision import WindowsFrameHandler
from hlrl.torch.vision.transforms import (
    Grayscale, ConvertDimensionOrder, StackDimension, Interpolate
)

from speedrunnersai.speedrunners.actor import Actor
from speedrunnersai.speedrunners.structures import Player, Match

class SpeedRunnersEnv(Env):
    """
    The SpeedRunners environment to be interacted with.
    """
    PROCESS_NAME = "speedrunners.exe"

    def __init__(self,
        episode_length: float,
        res_shape: Tuple[int, int],
        action_delay: float = 0,
        grayscale: bool = True,
        stacked_frames: int = 1,
        window_size: Optional[Tuple[int, int, int, int]] = None,
        device: str = "cpu",
        read_mem: bool = False):
        """
        Creates the environment.

        Args:
            episode_length (float): The maximum amount of time an agent can play
                before an environment reset.
            res_shape (Tuple[int, int]): The shape to resize the captured
                images to.
            action_delay (int): The amount of time in between actions, will wait
                for the delay before committing the action.
            grayscale (bool): Will convert the image to grayscale if true.
            stacked_frames (int): The number of frames to stack
                (along channel dimension)
            window_size (Optional[Tuple[int, int, int, int]]): The values of
                (left, top, right, bottom) to capture.
            device (str): The device to store tensors on.
            read_mem (bool): If true, will attach to the speedrunners process to
                read memory values (DO NOT USE ONLINE).
        """
        super().__init__()

        self.action_delay = action_delay
        self.state_space = (
            res_shape + ((1 if grayscale else 3) * stacked_frames,)
        )
        self.stacked_frames = stacked_frames
        self.window_size = window_size
        self.actor = Actor()
        self.last_window = None
        self.window = None

        # Create the d3dshot instance
        capture_output = (
            "pytorch_float" + ("_gpu" if str(device) == "cuda" else "")
        )
        d3d = d3dshot.create(
            capture_output=capture_output, frame_buffer_size=stacked_frames
        )

        frame_transforms = [
            lambda frame: frame.unsqueeze(0),
            ConvertDimensionOrder()
        ]

        if grayscale:
            frame_transforms.append(Grayscale())

        frame_transforms.append(Interpolate(size=res_shape, mode="bilinear"))
        frame_transforms.append(lambda frame: frame.squeeze(0))

        stack_transforms = [StackDimension(1)]

        self.frame_handler = WindowsFrameHandler(
            d3d, frame_transforms, stack_transforms
        )
        
        self.episode_length = episode_length
        self.action_space = (self.actor.num_actions(),)

        # The environment properties
        self._reached_goal = False

        # Relevant memory information
        self.memory = Pymem() if read_mem else None
        self.match = Match()
        self.match.players.append(Player())

        self.last_action_time = 0

        self._last_time = 0
        self.start_time = 0

    def reset(self) -> Any:
        """
        Resets the game (in practice).
        """
        self.actor.reset()
        self._episode_finished(True)

        for _ in range(self.stacked_frames):
            self.frame_handler.get_new_frame()

        if self.state is not None:
            del self.state

        self.state = self.frame_handler.get_frame_stack(
            range(self.stacked_frames), "first"
        )
        self.start_time = time()

        return self.state

    def start(self) -> None:
        """
        Starts the screen viewer.
        """
        self.frame_handler.capture(target_fps=240, region=self.window_size)

        # Make sure game window is on top
        self.window = win32gui.FindWindow(
            None, SpeedRunnersEnv.PROCESS_NAME.split(".")[0]
        )
        win32gui.SetForegroundWindow(self.window)

        if self.memory is not None:
            self.memory.open_process_from_name(SpeedRunnersEnv.PROCESS_NAME)

    def pause(self) -> None:
        """
        Closes the screen viewer.
        """
        self.frame_handler.stop()

    def stop(self) -> None:
        """
        Closes the environment.
        """
        self.reset()
        self.frame_handler.stop()
        self.actor.stop()

        if self.memory is not None:
            self.memory.close_process()

    def step(self, action: int) -> Tuple[Any, float, bool]:
        """
        Takes a step into the environment with the action given.

        Args:
            action (int): The index of the action to take.
        """
        # Only act if the game window is in the foreground, reset keyboard and
        # mouse on switch from game to something else as well
        window = win32gui.GetForegroundWindow()
        while window != self.window:
            if self.last_window == self.window:
                self.actor.reset()

            self.last_window = window

            window = win32gui.GetForegroundWindow()

        self.last_window = self.window

        # Wait until the delay has passed
        current_time = time()
        while current_time - self.last_action_time < self.action_delay:
            current_time = time()

        self.last_action_time = current_time

        self.actor.act(action)

        self.frame_handler.get_new_frame()

        if self.state is not None:
            del self.state

        self.state = self.frame_handler.get_frame_stack(
            range(self.stacked_frames), "first"
        )

        self._get_reward()
        self._episode_finished(False)

        return self.state, self.reward, self.terminal, None

    def sample_action(self) -> int:
        """
        Returns a random action in the environment.
        """
        return self.actor.sample_action()

    def render(self) -> None:
        """
        Here to not break certain agents, nothing needs to be done to render.
        """
        pass

    def _update_memory(self) -> None:
        """
        Updates match with the new values of the process.
        """
        if self.memory is not None:
            self.match.update(self.memory, 0)

    def _get_reward(self) -> None:
        """
        Returns the reward for the current state of the environment.
        """
        # Update structures with new memory
        self._update_memory()

        # REWARD IS FOR SWING ACROSS MAP
        player = self.match.players[0]
        reward = np.power(1.01, player.speed() / 3) / 150

        # Position starts at the top left, want to encourage going higher
        reward -= (player.vertical_position - 1250) / 1000

        if self._reached_goal:
            reward += 10

        self.reward = reward

    def _get_obstacles_hit(self) -> int:
        """
        return (self.memory.get_address(self.addresses["obstacles_hit"],
                                        ctypes.c_byte)
                if self.memory is not None else 0)
        """
        return 0

    def _episode_finished(self, reset: bool = False) -> None:
        if reset:
            self._last_time = 0
            self.terminal = False
            self._reached_goal = False
        else:
            if(self.match.current_time < self._last_time):
                self._last_time = 0
                self.terminal = True
                self._reached_goal = True
            elif((self.episode_length is not None
                and self.match.current_time > self.episode_length)
                or (self.episode_length is not None
                    and (time() - self.start_time) > self.episode_length)):
                self._last_time = 0
                self.terminal = True
                self._reached_goal = False
            else:
                self._last_time = self.match.current_time
                self.terminal = False
                self._reached_goal = False
