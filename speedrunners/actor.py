from time import sleep

from pykeyboard import PyKeyboard

class Actor():
    def __init__(self):
        # Get the keyboard
        self.keyboard = PyKeyboard()

        # The reset button
        self._reset = "r"

        # All the keys for the game configuration
        self._jump = "a"
        self._grapple = "s"
        self._item = "d"
        self._slide = self.keyboard.down_key
        self._left = self.keyboard.left_key
        self._right = self.keyboard.right_key
        self._boost = self.keyboard.space_key

        # All the keys being used
        self._keys = [self._jump, self._grapple, self._item, self._slide,
                      self._left, self._right, self._boost]

        # The last action that was taken
        self._last_command = ""

        # All of the possible actions
        self._actions = {0 : self.left,
                         1 : self.left_boost,
                         2 : self.right,
                         3 : self.right_boost,
                         4 : self.jump,
                         5 : self.jump_left,
                         6 : self.jump_left_boost,
                         7 : self.jump_right,
                         8 : self.jump_right_boost,
                         9 : self.grapple,
                         10 : self.grapple_left,
                         11 : self.grapple_right,
                         12 : self.item,
                         13 : self.item_boost,
                         14 : self.item_left,
                         15 : self.item_left_boost,
                         16 : self.item_right,
                         17 : self.item_right_boost,
                         18 : self.slide}

        # The number of possible actinos
        self.num_actions = len(self._actions)

    def perform_action(self, action):
        self._actions[action]()

    def reset(self):
        self.keyboard.press_key(self._reset)
        sleep(1e-2)
        self.keyboard.release_key(self._reset)

        self.release_keys()

    def press_keys(self, *keys):
        # Loop through all the keys
        for key in self._keys:
            # Press the key if it was given
            if(key in keys):
                self.keyboard.press_key(key)

            # Release it otherwise
            else: 
                self.keyboard.release_key(key)

    def left(self):
        if(self._last_command != "left"):
            self.press_keys(self._left)

            self._last_command = "left"

    def left_boost(self):
        if(self._last_command != "left_boost"):
            self.press_keys(self._left, self._boost)

            self._last_command = "left_boost"

    def right(self):
        if(self._last_command != "right"):
            self.press_keys(self._right)

            self._last_command = "right"

    def right_boost(self):
        if(self._last_command != "right_boost"):
            self.press_keys(self._right, self._boost)

            self._last_command = "right_boost"

    def jump(self):
        if(self._last_command != "jump"):
            self.press_keys(self._jump)

            self._last_command = "jump"

    def jump_left(self):
        if(self._last_command != "jump_left"):
            self.press_keys(self._jump, self._left)

            self._last_command = "jump_left"

    def jump_left_boost(self):
        if(self._last_command != "jump_left_boost"):
            self.press_keys(self._jump, self._left, self._boost)
            
            self._last_command = "jump_left_boost"


    def jump_right(self):
        if(self._last_command != "jump_right"):
            self.press_keys(self._jump, self._right)

            self._last_command = "jump_right"

    def jump_right_boost(self):
        if(self._last_command != "jump_right_boost"):
            self.press_keys(self._jump, self._right, self._boost)

            self._last_command = "jump_right_boost"

    def grapple(self):
        if(self._last_command != "grapple"):
            self.press_keys(self._grapple)

            self._last_command = "grapple"

    def grapple_left(self):
        if(self._last_command != "grapple_left"):
            self.press_keys(self._grapple, self._left)

            self._last_command = "grapple_left"

    def grapple_right(self):
        if(self._last_command != "grapple_right"):
            self.press_keys(self._grapple, self._right)

            self._last_command = "grapple_right"

    def item(self):
        if(self._last_command != "item"):
            self.press_keys(self._item)

            self._last_command = "item"

    def item_boost(self):
        if(self._last_command != "item_boost"):
            self.press_keys(self._item, self._boost)

            self._last_command = "item_boost"

    def item_left(self):
        if(self._last_command != "item_left"):
            self.press_keys(self._item, self._left)

            self._last_command = "item_left"

    def item_left_boost(self):
        if(self._last_command != "item_left_boost"):
            self.press_keys(self._item, self._left, self._boost)

            self._last_command = "item_left_boost"

    def item_right(self):
        if(self._last_command != "item_right"):
            self.press_keys(self._item, self._right)

            self._last_command = "item_right"

    def item_right_boost(self):
        if(self._last_command != "item_right_boost"):
            self.press_keys(self._item, self._right, self._boost)

            self._last_command = "item_right_boost"

    def slide(self):
        if(self._last_command != "slide"):
            self.press_keys(self._slide)

            self._last_command = "slide"

    def release_keys(self):
        for key in self._keys:
            self.keyboard.release_key(key)
