# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pidm_imitation.utils.user_inputs import UserInputs


class AxisMapping:
    """Defines the joystick index for a given axis, and whether to ignore a given deadzone
    and whether to scale the floating point values"""

    def __init__(self, index: int, deadzone: float = 0.0, scale: float = 1.0):
        self.index = index
        self.scale = scale
        self.deadzone = deadzone


class JoystickMap:
    """Contains the mapping for buttons, hats and axes from the pygame joystick"""

    def __init__(self):
        self.buttons = {}
        self.hats = {}
        self.axes = {}
        self.reported_error = False

    @staticmethod
    def load_from_config(name, config: dict) -> "JoystickMap":
        if name in config:
            map = JoystickMap()
            mapping: dict = config[name]
            map.buttons = mapping["buttons"]
            map.hats = mapping["hats"]
            map.axes = mapping["axes"]
            for key in map.axes:
                s = map.axes[key]
                map.axes[key] = JoystickMap.parse_axis(s)
            return map
        return None

    @staticmethod
    def parse_axis(s: str) -> AxisMapping:
        args = s.strip("AxisMapping").strip("(").strip(")").split(",")
        if len(args) < 1:
            raise Exception("Unexpected number of arguments")
        v = [float(x) for x in args]
        index = int(v[0])
        if len(v) == 1:
            return AxisMapping(index)
        elif len(v) == 2:
            return AxisMapping(index, v[1])
        elif len(v) == 3:
            return AxisMapping(index, v[1], v[2])
        raise Exception("Unexpected number of arguments")

    def get_mapped_button_value(self, joystick, name, default_value=0) -> int:
        """Find out if the named input is in the buttons map and if so
        get that mapped button from the joystick, otherwise return the
        default_value"""
        if name in self.buttons:
            return joystick.get_button(self.buttons[name])
        return default_value

    def get_mapped_axis_value(self, joystick, name, default_value=0) -> float:
        """Find out if the named input is in the axes map and if so
        get that mapped axis from the joystick, otherwise return the
        default_value"""
        if name in self.axes:
            mapping: AxisMapping = self.axes[name]
            v = joystick.get_axis(mapping.index)
            if abs(v) < mapping.deadzone:
                return 0
            return v * mapping.scale
        return float(default_value)

    def boolean_vector(self, a, b) -> int:
        """Map two button values to the vector (-1, 0, 1)"""
        if a:
            return -1
        if b:
            return 1
        return 0

    def get_user_inputs(self, joystick) -> UserInputs:
        """Map the current button and axis values from the given joystick"""
        result = UserInputs()
        result.a = self.get_mapped_button_value(joystick, "a")
        result.b = self.get_mapped_button_value(joystick, "b")
        result.x = self.get_mapped_button_value(joystick, "x")
        result.y = self.get_mapped_button_value(joystick, "y")

        result.left_bumper = self.get_mapped_button_value(joystick, "left_bumper")
        result.right_bumper = self.get_mapped_button_value(joystick, "right_bumper")
        result.left_stick = self.get_mapped_button_value(joystick, "left_stick")
        result.right_stick = self.get_mapped_button_value(joystick, "right_stick")
        result.guide = 0  # TODO
        result.view = self.get_mapped_button_value(joystick, "view")
        result.menu = self.get_mapped_button_value(joystick, "menu")
        result.left_stick_x = self.get_mapped_axis_value(joystick, "left_stick_x")
        result.left_stick_y = self.get_mapped_axis_value(joystick, "left_stick_y")
        result.right_stick_x = self.get_mapped_axis_value(joystick, "right_stick_x")
        result.right_stick_y = self.get_mapped_axis_value(joystick, "right_stick_y")
        result.left_trigger = self.get_mapped_axis_value(joystick, "left_trigger")
        result.right_trigger = self.get_mapped_axis_value(joystick, "right_trigger")

        result.dpad_x = 0
        result.dpad_y = 0
        if "dpad" in self.hats:
            if joystick.get_numhats() > 0:
                dpad = self.hats["dpad"]
                result.dpad_x, result.dpad_y = joystick.get_hat(dpad)
            elif not self.reported_error:
                print(
                    "### Warning: dpad mapping is set but no hats were found on the pygame joystick object"
                )
                self.reported_error = True
        else:
            dpad_left = self.get_mapped_button_value(joystick, "dpad_left")
            dpad_right = self.get_mapped_button_value(joystick, "dpad_right")
            dpad_up = self.get_mapped_button_value(joystick, "dpad_up")
            dpad_down = self.get_mapped_button_value(joystick, "dpad_down")
            result.dpad_x = self.boolean_vector(dpad_left, dpad_right)
            result.dpad_y = self.boolean_vector(dpad_down, dpad_up)
        return result


class XBoxGamePadMap(JoystickMap):
    """This is a JoystickMap for the XBox Gamepad."""

    def __init__(self):
        self.buttons = {
            "a": 0,
            "b": 1,
            "x": 2,
            "y": 3,
            "left_bumper": 4,
            "right_bumper": 5,
            "view": 6,
            "menu": 7,
            "left_stick": 8,
            "right_stick": 9,
        }
        self.hats = {"dpad": 0}
        self.axes = {
            "left_stick_x": AxisMapping(0),
            "left_stick_y": AxisMapping(1, 0, -1.0),
            "right_stick_x": AxisMapping(2),
            "right_stick_y": AxisMapping(3, 0, -1.0),
            "left_trigger": AxisMapping(4),
            "right_trigger": AxisMapping(5),
        }
