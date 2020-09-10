import numpy as np

#from pykeyboard import PyKeyboard
from collections import OrderedDict
from configparser import ConfigParser

class Actor():
    one_hot_acts = np.eye(17)

    ACTIONS = {"left" : one_hot_acts[0],
               "left_boost" : one_hot_acts[1],
               "right" : one_hot_acts[2],
               "right_boost" : one_hot_acts[3],
               "left_jump" : one_hot_acts[4],
               "left_jump_boost" : one_hot_acts[5],
               "right_jump" : one_hot_acts[6],
               "right_jump_boost" : one_hot_acts[7],
               "left_grapple" : one_hot_acts[8],
               "left_grapple_boost" : one_hot_acts[9],
               "right_grapple" : one_hot_acts[10],
               "right_grapple_boost" : one_hot_acts[11],
               "left_item" : one_hot_acts[12],
               "left_item_boost" : one_hot_acts[13],
               "right_item" : one_hot_acts[14],
               "right_item_boost" : one_hot_acts[15],
               "slide" : one_hot_acts[16]
              }

    CONT_ACTIONS = {0 : [0, 0, 0, 0, 0, 1, 0],
                    1 : [0, 0, 0, 1, 0, 1, 0],
                    2 : [0, 0, 0, 0, 0, 0, 1],
                    3 : [0, 0, 0, 1, 0, 0, 1],
                    4 : [1, 0, 0, 0, 0, 1, 0],
                    5 : [1, 0, 0, 1, 0, 1, 0],
                    6 : [1, 0, 0, 0, 0, 0, 1],
                    7 : [1, 0, 0, 1, 0, 0, 1],
                    8 : [0, 1, 0, 0, 0, 1, 0],
                    9 : [0, 1, 0, 1, 0, 1, 0],
                    10 : [0, 1, 0, 0, 0, 0, 1],
                    11 : [0, 1, 0, 1, 0, 0, 1],
                    12 : [0, 0, 1, 0, 0, 1, 0],
                    13 : [0, 0, 1, 1, 0, 1, 0],
                    14 : [0, 0, 1, 0, 0, 0, 1],
                    15 : [0, 0, 1, 1, 0, 0, 1],
                    16 : [0, 0, 0, 0, 1, 0, 0],
                    }

    def __init__(self):
        """
        Creates an actor for the game speedrunners
        """
        # Get the keyboard
        self.keyboard = PyKeyboard()

        # Read the player key-bindings from the config
        self.speedrunners = self.read_config()

        if(self.speedrunners["BOOST"] == ""):
            self.speedrunners["BOOST"] = " "

        # The current values of the actions and the corresponding function
        self.action_values = [(self.speedrunners["JUMP"], 0),
                              (self.speedrunners["GRAPPLE"], 0),
                              (self.speedrunners["ITEM"], 0),
                              (self.speedrunners["BOOST"], 0),
                              (self.speedrunners["SLIDE"], 0),
                              (self.speedrunners["LEFT"], 0),
                              (self.speedrunners["RIGHT"], 0)]

        self._reset = self.speedrunners["RESET"]

        self.action_values = OrderedDict(self.action_values)
        self.action_values_items = list(self.action_values.items())

    @staticmethod
    def num_actions():
        """
        Returns the number of actions the actor can take.
        """
        return len(ACTIONS)

    def act(self, actions):
        """
        Causes the actor to act based on the actions given

        action : The actions for the actor to perform, a (6,) size array
        """
        actions = CONT_ACTIONS[actions]
        for i in range(len(actions)):
            # Get the value of the action
            value = actions[i]

            # Get the action
            action = self.action_values_items[i][0]

            # Start or stop the action
            self.set_action(action, value)

    def set_action(self, action, value):
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

    def single_action(self, action):
        """
        Does the single action given without checking for other actions

        action : The action to perform
        """
        # Get the current value of the action
        value = self.action_values[action]

        # Check if the action is already down
        if(value != 1):
            if(action.lower() == "down"):
                self.keyboard.press_key(self.keyboard.down_key)
            elif(action.lower() == "left"):
                self.keyboard.press_key(self.keyboard.left_key)
            elif(action.lower() == "right"):
                self.keyboard.press_key(self.keyboard.right_key)
            elif(action.lower() == "space"):
                self.keyboard.press_key(self.keyboard.space_key)
            else:
                self.keyboard.press_key(action)

            self.action_values[action] = 1

    def stop_action(self, action, total_reset = False):
        """
        Stops performing the given action if it is down

        action : The action to stop performing
        total_reset : If both direction keys should be released if given a
                      direction key
        """
        # Get the current value of the action
        value = self.action_values[action]

        # Check if the action is already down
        if(value != 0):
            if(action.lower() == "down"):
                self.keyboard.release_key(self.keyboard.down_key)
            elif(action.lower() == "left"):
                self.keyboard.release_key(self.keyboard.left_key)
            elif(action.lower() == "right"):
                self.keyboard.release_key(self.keyboard.right_key)
            elif(action.lower() == "space"):
                self.keyboard.release_key(self.keyboard.space_key)
            else:
                self.keyboard.release_key(action)

            self.action_values[action] = 0

    def release_keys(self):
        """
        Releases every key in the current list of given actions
        """
        for key in self.action_values:
            self.stop_action(key)

    def reset(self):
        """
        Resets the game.
        """
        # Tap not working
        self.keyboard.press_key(self._reset)
        
        self.release_keys()

    def stop(self):
        """
        Closes the actor.
        """
        self.release_keys()

    def continuous_to_discrete(self, action):
        """
        Converts an array of actions in the continuous form to a discrete action

        action : The action to convert
        """
        key = ""

        if(action[4]):
            key = "slide"
        else:
            if(action[5]):
                key += "left"
            else:
                key += "right"

            if(action[0]):
                key += "_jump"
            elif(action[1]):
                key += "_grapple"
            elif(action[2]):
                key += "_item"

            if(action[3]):
                key += "_boost"

        return ACTIONS[key]

    def read_config(self):
        """
        Reads the config file to obtain the settings for the recorder, the
        window size for the training data and the game bindings
        """
        config = ConfigParser()
    
        # Read the config file, make sure not to re-name
        config.read("./config/config.ini")

        return config["SpeedRunners Config"]                                                  
