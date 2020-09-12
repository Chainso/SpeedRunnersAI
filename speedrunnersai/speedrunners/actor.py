import numpy as np
import keyboard

from collections import OrderedDict
from configparser import ConfigParser

class Actor():
    one_hot_acts = np.eye(17)

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
        ("left_item", [0, 0, 1, 0, 0, 1, 0]),
        ("left_item_boost", [0, 0, 1, 1, 0, 1, 0]),
        ("right_item", [0, 0, 1, 0, 0, 0, 1]),
        ("right_item_boost", [0, 0, 1, 1, 0, 0, 1]),
        ("slide", [0, 0, 0, 0, 1, 0, 0]),
    ])

    ACTION_ITEMS = list(ACTIONS.items())

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

        self.action_values = [(self.speedrunners["JUMP"], 0),
                              (self.speedrunners["GRAPPLE"], 0),
                              (self.speedrunners["ITEM"], 0),
                              (self.speedrunners["BOOST"], 0),
                              (self.speedrunners["SLIDE"], 0),
                              (self.speedrunners["LEFT"], 0),
                              (self.speedrunners["RIGHT"], 0)]

        self._reset = self.speedrunners["RESET"]

    @staticmethod
    def num_actions():
        """
        Returns the number of actions the actor can take.
        """
        return len(Actor.ACTIONS)

    def act(self, action: int):
        """
        Causes the actor to act based on the action given.

        action (int): The action for the actor to perform.
        """
        keys = Actor.ACTION_ITEMS[action][-1]
        for i in range(len(keys)):
            self.set_action(i, keys[i])

    def set_action(self, action: int, value: int):
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
        # Tap not working
        self.release_keys()
        keyboard.press(self._reset)

    def stop(self):
        """
        Closes the actor.
        """
        self.release_keys()

    def continuous_to_discrete(self, action):
        """
        Converts an array of actions in the continuous form to a discrete
        action.

        action : The action to convert
        """
        return Actor.ACTIONS.keys().index(action)

    def sample_action(self) -> int:
        """
        Returns a random action.
        """
        return np.random.randint(0, self.num_actions())

    def read_config(self):
        """
        Reads the config file to obtain the SpeedRunners key bindings.
        """
        config = ConfigParser()
    
        # Read the config file, make sure not to re-name
        config.read("../config/config.ini")

        return config["SpeedRunners Config"]                                                  
