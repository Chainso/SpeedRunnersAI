import numpy as np
import keyboard

from typing import Tuple
from collections import OrderedDict
from configparser import ConfigParser
from os import path
from time import sleep

class ContinuousActor():
    """
    The actor to input actions to the game, using continuous values in (-1, 1)
    """   
    def __init__(self):
        """
        Creates an actor for the game speedrunners
        """
        # Read the player key-bindings from the config
        self.speedrunners = self.read_config()

        # The current values of the actions and the corresponding function
        self.action_keys = [
            self.speedrunners["JUMP"],
            self.speedrunners["GRAPPLE"],
            self.speedrunners["ITEM"],
            self.speedrunners["BOOST"],
            self.speedrunners["SLIDE"],
            self.speedrunners["LEFT"],
            self.speedrunners["RIGHT"]
        ]

        self.reset_key = self.speedrunners["RESET"]

    @staticmethod
    def num_actions() -> int:
        """
        Returns the number of actions the actor can take.
        """
        return len(self.action_keys)

    def act(self, action: Tuple[float]) -> None:
        """
        Causes the actor to act based on the action given.

        action (Tuple[float]): A tuple of values in (-1, 1) representation the
            on/off state of each key.
        """
        for i in range(len(action)):
            self.set_action(i, action[i])

    def set_action(self, action: int, value: int) -> None:
        """
        Will perform the action and set the value to 1 if the given value is
        positive, otherwise will stop performing the action and set the value
        to -1 if the given value is negative.

        action : The action to either stop or perform.
        value : The value of the action, negative to stop, positive to perform.
        """
        # Check the value
        if(value == 0):
            self.stop_action(action)
        else:
            self.single_action(action)

    def single_action(self, action: int):
        """
        Does the single action given without checking for other actions.

        action : The action to perform.
        """
        keyboard.press(self.action_keys[action])

    def stop_action(self, action):
        """
        Stops performing the given action if it is down

        action : The action to stop performing
        """
        keyboard.release(self.action_keys[action])

    def release_keys(self):
        """
        Releases every key in the current list of given actions.
        """
        for key in self.action_keys:
            keyboard.release(key)

    def reset(self):
        """
        Resets the game.
        """
        self.release_keys()

        # Tap (press_and_release) not working
        # Try to find a viable solution without sleep
        keyboard.press(self.reset_key)
        sleep(1)
        keyboard.release(self.reset_key)

    def stop(self):
        """
        Closes the actor.
        """
        self.release_keys()

    def continuous_to_discrete(self, action: Tuple[float]) -> int:
        """
        Converts an array of actions in the continuous form to a discrete
        action.

        action (Tuple[float]): The action to convert.
        """
        return Actor.ACTIONS.keys().index(action)

    def sample_action(self) -> Tuple[float]:
        """
        Returns a random action.
        """
        return 2 * np.random.rand(self.num_actions()) - 1

    def read_config(self):
        """
        Reads the config file to obtain the SpeedRunners key bindings.
        """
        config = ConfigParser()
    
        # Read the config file, make sure not to re-name
        config_path = path.dirname(path.abspath(__file__))
        config_path += "/../config/config.ini"

        config.read(config_path)

        return config["SpeedRunners Config"]                                                  

class Actor(ContinuousActor):
    """
    The actor for the game using a preset of discrete action combinations.
    """
    ACTIONS = OrderedDict([
        ("left", [0, 0, 0, 0, 0, 1, 0]),
        ("left_boost", [0, 0, 0, 1, 0, 1, 0]),
        ("right", [0, 0, 0, 0, 0, 0, 1]),
        ("right_boost", [0, 0, 0, 1, 0, 0, 1]),
        ("left_jump", [1, 0, 0, 0, 0, 1, 0]),
        ("left_jump_boost", [1, 0, 0, 1, 0, 1, 0]),
        ("right_jump", [1, 0, 0, 0, 0, 0, 1]),
        ("right_jump_boost", [1, 0, 0, 1, 0, 0, 1]),
        ("left_grapple", [0, 1, 0, 0, 0, 1, 0]),
        ("left_grapple_boost", [0, 1, 0, 1, 0, 1, 0]),
        ("right_grapple", [0, 1, 0, 0, 0, 0, 1]),
        ("right_grapple_boost", [0, 1, 0, 1, 0, 0, 1]),
        #("left_item", [0, 0, 1, 0, 0, 1, 0]),
        #("left_item_boost", [0, 0, 1, 1, 0, 1, 0]),
        #("right_item", [0, 0, 1, 0, 0, 0, 1]),
        #("right_item_boost", [0, 0, 1, 1, 0, 0, 1]),
        ("slide", [0, 0, 0, 0, 1, 0, 0]),
    ])

    ACTION_ITEMS = list(ACTIONS.items())

    @staticmethod
    def num_actions() -> int:
        """
        Returns the number of actions the actor can take.
        """
        return len(Actor.ACTIONS)

    def act(self, action: int) -> None:
        """
        Causes the actor to act based on the action given.

        action (int): The discrete action to take.
        """
        keys = Actor.ACTION_ITEMS[action][-1]
        super().act(keys)

    def sample_action(self) -> int:
        """
        Returns a random action.
        """
        return np.random.randint(0, self.num_actions())