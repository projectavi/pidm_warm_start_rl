# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations

import json
from typing import List

import numpy as np

# Camera actions are in degrees, while mouse movement is in pixels
# Multiply mouse speed by some arbitrary multiplier
MOUSE_MULTIPLIER = 0.01
STICK_DEAD_ZONE = 27  # there is a right joy stick dead zone, any less than that and camera stops moving.


class UserInputs:
    """This is a serializable container for a user input with a timestamp so that a
    recording can be played back at the same speed and it can map those inputs to the
    bleeding edge came gRPC actions"""

    def __init__(
        self,
        ticks=0,
        a=0,
        x=0,
        y=0,
        b=0,
        left_bumper=0,
        right_bumper=0,
        guide=0,
        view=0,
        menu=0,
        left_stick=0,
        right_stick=0,
        dpad_x=0,  # -1=left, 0=none, 1=right
        dpad_y=0,  # 1=up, 0=none, -1=down
        left_trigger=0.0,
        right_trigger=0.0,
        left_stick_x=0.0,
        left_stick_y=0.0,
        right_stick_x=0.0,
        right_stick_y=0.0,
        keys_down=[],
        mouse=(0, 0),
    ):
        self.ticks = ticks
        self.a = a
        self.x = x
        self.y = y
        self.b = b
        self.left_bumper = left_bumper
        self.right_bumper = right_bumper
        self.guide = guide
        self.view = view
        self.menu = menu
        self.left_stick = left_stick  # buttons
        self.right_stick = right_stick
        self.dpad_x = dpad_x
        self.dpad_y = dpad_y
        self.left_trigger = left_trigger
        self.right_trigger = right_trigger
        self.left_stick_x = left_stick_x
        self.left_stick_y = left_stick_y
        self.right_stick_x = right_stick_x
        self.right_stick_y = right_stick_y
        self.keys_down = keys_down
        self.mouse = mouse

    def is_same(self, other):
        """This method uses numpy all_close so that almost identical floating point values
        are considered equal, this reduces the size of the recording files"""
        return (
            self.a == other.a
            and self.x == other.x
            and self.y == other.y
            and self.b == other.b
            and self.left_bumper == other.left_bumper
            and self.right_bumper == other.right_bumper
            and self.guide == other.guide
            and self.view == other.view
            and self.menu == other.menu
            and self.left_stick == other.left_stick
            and self.right_stick == other.right_stick
            and self.dpad_x == other.dpad_x
            and self.dpad_y == other.dpad_y
            and self.keys_down == other.keys_down
            and np.allclose(self.mouse, other.mouse)
            and np.allclose(
                [
                    self.left_trigger,
                    self.right_trigger,
                    self.left_stick_x,
                    self.left_stick_y,
                    self.right_stick_x,
                    self.right_stick_y,
                ],
                [
                    other.left_trigger,
                    other.right_trigger,
                    other.left_stick_x,
                    other.left_stick_y,
                    other.right_stick_x,
                    other.right_stick_y,
                ],
            )
        )

    def is_button_a(self):
        return self.a != 0 or "space" in self.keys_down

    def is_button_b(self):
        return self.b != 0 or "b" in self.keys_down

    def is_button_y(self):
        return self.y != 0 or "y" in self.keys_down

    def is_button_x(self):
        return self.x != 0 or "x" in self.keys_down

    def get_left_stick_x(self):
        if "a" in self.keys_down:
            return -1
        elif "d" in self.keys_down:
            return 1
        return self.left_stick_x

    def get_left_stick_y(self):
        if "w" in self.keys_down:
            return 1
        elif "s" in self.keys_down:
            return -1
        return self.left_stick_y

    def get_right_trigger(self):
        return self.right_trigger

    def get_left_trigger(self):
        return self.left_trigger

    def map_mouse_to_stick(self, x):
        x = (x + np.sign(x) * STICK_DEAD_ZONE) * MOUSE_MULTIPLIER
        if x < -1:
            return -1
        elif x > 1:
            return 1
        return x

    def get_right_stick_x(self):
        dx = self.mouse[0]
        if dx != 0:
            return self.map_mouse_to_stick(dx)
        return self.right_stick_x

    def get_right_stick_y(self):
        dy = self.mouse[1]
        if dy != 0:
            return self.map_mouse_to_stick(dy)
        return self.right_stick_y

    def get_button_state(self, button: str) -> bool:
        if button == "left":
            return self.dpad_x < 0
        elif button == "right":
            return self.dpad_x > 0
        elif button == "up":
            return self.dpad_y > 0
        elif button == "down":
            return self.dpad_y < 0
        elif hasattr(self, button):
            return getattr(self, button) != 0
        else:
            assert False, f"Cannot get invalid button {button}"

    def set_button_state(self, button: str, state: bool):
        if button == "left":
            self.dpad_x = -1 if state else 0
        elif button == "right":
            self.dpad_x = 1 if state else 0
        elif button == "up":
            self.dpad_y = 1 if state else 0
        elif button == "down":
            self.dpad_y = -1 if state else 0
        elif hasattr(self, button):
            setattr(self, button, 1 if state else 0)
        else:
            assert False, f"Cannot set invalid button {button}"

    def get_button_state_dict(self):
        """Must also match the serialization format for SmartReplayApp."""
        return {
            "a": self.is_button_a(),
            "x": self.is_button_x(),
            "y": self.is_button_y(),
            "b": self.is_button_b(),
            "left_bumper": self.left_bumper,
            "right_bumper": self.right_bumper,
            "guide": self.guide,
            "view": self.view,
            "menu": self.menu,
            "left_stick": self.left_stick,
            "right_stick": self.right_stick,
            "left": 1 if self.dpad_x < 0 else 0,
            "right": 1 if self.dpad_x > 0 else 0,
            "up": 1 if self.dpad_y > 0 else 0,
            "down": 1 if self.dpad_y < 0 else 0,
        }

    def get_gamepad_state_dict(self):
        """Must match the serialization format for SmartReplayApp.GamepadEvent"""
        return {
            "a": self.a,
            "x": self.x,
            "y": self.y,
            "b": self.b,
            "left_bumper": self.left_bumper,
            "right_bumper": self.right_bumper,
            "guide": self.guide,
            "view": self.view,
            "menu": self.menu,
            "left_stick": self.left_stick,
            "right_stick": self.right_stick,
            "left_stick_x": self.left_stick_x,
            "left_stick_y": self.left_stick_y,
            "right_stick_x": self.right_stick_x,
            "right_stick_y": self.right_stick_y,
            "left_trigger": self.left_trigger,
            "right_trigger": self.right_trigger,
            "dpad_left": 1 if self.dpad_x < 0 else 0,
            "dpad_right": 1 if self.dpad_x > 0 else 0,
            "dpad_up": 1 if self.dpad_y > 0 else 0,
            "dpad_down": 1 if self.dpad_y < 0 else 0,
        }


class UserInputsLog:
    """Stores all UserInputs for a session so you can save them to a file and
    load them back in later during replay"""

    def __init__(self):
        self.inputs: List[UserInputs] = []

    def record(self, action: UserInputs):
        """Record a new user input"""
        self.inputs.append(action)

    def size(self):
        """Return the number of recorded inputs"""
        return len(self.inputs)

    def save(self, filename: str):
        """Save all recorded inputs to disk as json"""
        first = True
        with open(filename, "w") as f:
            f.write("[")
            for a in self.inputs:
                if not first:
                    f.write(",")
                json.dump(a.__dict__, f)
                f.write("\n")
                first = False
            f.write("]")

    def load(self, filename):
        """Load a previously saved json file back into the list of recorded inputs"""
        self.inputs = []
        with open(filename, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                for d in data:
                    self.record(UserInputs(**d))
            else:
                raise Exception(f"Expecting log file '{filename}' to contain a json array")


class VideoTicks:
    """Stores an array of video ticks for a recording"""

    TICKS_KEY = "video_ticks"
    INVALID_DURATION = -1
    INVALID_FPS = -1

    def __init__(self):
        self.video_ticks: List[float] = []

    def record(self, tick: float):
        """Record a new user input"""
        self.video_ticks.append(tick)

    @property
    def frame_count(self) -> int:
        return len(self.video_ticks)

    @property
    def duration(self) -> float:
        return (
            (self.video_ticks[-1] - self.video_ticks[0]) if len(self.video_ticks) > 0 else VideoTicks.INVALID_DURATION
        )

    @property
    def video_fps(self) -> float:
        return (
            (len(self.video_ticks) - 1) / (self.video_ticks[-1] - self.video_ticks[0])
            if len(self.video_ticks) > 1
            else VideoTicks.INVALID_FPS
        )

    def save(self, filename: str):
        """Save all recorded video ticks to disk as json"""
        with open(filename, "w") as f:
            serialized_videologs = json.dumps(self.__dict__)
            f.write(serialized_videologs)

    @staticmethod
    def load(filename: str) -> VideoTicks:
        """Load a previously saved json file back into VideoTicks"""
        with open(filename, "r") as f:
            data = json.load(f)
            video_ticks = VideoTicks()
            if VideoTicks.TICKS_KEY not in data:
                raise ValueError(
                    f"Invalid json provided for VideoTicks. \
                                    The file should have the following field: {VideoTicks.TICKS_KEY}."
                )
            video_ticks.video_ticks = data[VideoTicks.TICKS_KEY]
            return video_ticks
