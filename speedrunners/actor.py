from pykeyboard import PyKeyboard
from collections import OrderedDict
from configparser import ConfigParser

class Actor():
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

    def act(self, actions):
        """
        Causes the actor to act based on the actions given

        action : The actions for the actor to perform, a (6,) size array
        """
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

    def num_actions(self):
        """
        Returns the number of actions the actor can take.
        """
        return len(self.action_values)

    def read_config(self):
        """
        Reads the config file to obtain the settings for the recorder, the
        window size for the training data and the game bindings
        """
        config = ConfigParser()
    
        # Read the config file, make sure not to re-name
        config.read("config.ini")

        return config["SpeedRunners Config"]                                                  
