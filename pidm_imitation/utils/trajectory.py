# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from pidm_imitation.constants import (
    INPUTS_FILE_SUFFIX,
    TRAJECTORY_DATA_FILE_SUFFIX,
    VIDEO_FILE_SUFFIX,
    VIDEO_METADATA_SUFFIX,
)
from pidm_imitation.utils.ioutils import read_video, save_video
from pidm_imitation.utils.logger import Logger
from pidm_imitation.utils.user_inputs import UserInputs, UserInputsLog, VideoTicks

log = Logger.get_logger(__name__)


class Trajectory:
    """This class represents a combination of optional user_actions that are needed to drive
    along a given trajectory, along with option video frames, and timing information for when
    each of those video frames was recorded (in seconds from the start of the recording)
    """

    VIDEO_KEY = "video"
    VIDEO_TICKS_KEY = "video_ticks"
    INPUTS_KEY = "inputs"
    DATA_KEY = "data"
    LOAD_SAVE_KEYS = [VIDEO_KEY, VIDEO_TICKS_KEY, INPUTS_KEY, DATA_KEY]

    def __init__(
        self,
        video_source: str | None = None,
        video_frames: List[np.ndarray] | None = None,
        video_ticks: VideoTicks | None = None,
        user_inputs: UserInputsLog | None = None,
        user_input_ticks: List[float] | None = None,
        video_fps: int | None = None,
        color_format: str = "bgr",
    ):
        """
        :param video_source: The source filename of the video recorded during the trajectory.
        :param video_frames: The video frames recorded during the trajectory. Should be
            list of frames with shape (width, height, 3).
        :param video_ticks: The VideoTicks object containing timing information for the video frames.
        :param user_inputs: The UserInputsLog object containing user inputs during the trajectory.
        :param user_input_ticks: The list of timestamps (in seconds) for each user input.
        :param video_fps: The frames per second of the video. If None, it will be inferred from video_ticks.
        :param color_format: The color format of the video frames, either 'bgr' or 'rgb'.
        """
        self.video_source = video_source
        self.video_frames = video_frames if video_frames is not None else []
        self.video_fps = (
            int(video_ticks.video_fps)
            if video_ticks is not None and video_ticks.video_fps > 0
            else video_fps
        )
        self.video_ticks = video_ticks if video_ticks else VideoTicks()
        self.user_inputs = user_inputs
        self.user_input_ticks = user_input_ticks if user_input_ticks else []
        self.other_data: Dict[str, List[np.ndarray]] = {}
        self.color_format = color_format

        if self.user_inputs is None:
            self.user_inputs = UserInputsLog()

    @staticmethod
    def get_video_path(dirname: str, trajectory_name: str) -> str:
        return os.path.join(dirname, trajectory_name + VIDEO_FILE_SUFFIX + ".mp4")

    @staticmethod
    def get_video_ticks_path(dirname: str, trajectory_name: str) -> str:
        return os.path.join(dirname, trajectory_name + VIDEO_METADATA_SUFFIX + ".json")

    @staticmethod
    def get_user_inputs_path(dirname: str, trajectory_name: str) -> str:
        return os.path.join(dirname, trajectory_name + INPUTS_FILE_SUFFIX + ".json")

    @staticmethod
    def get_data_path(dirname: str, trajectory_name: str) -> str:
        return os.path.join(
            dirname, trajectory_name + TRAJECTORY_DATA_FILE_SUFFIX + ".npz"
        )

    @classmethod
    def init_from_dir(
        cls,
        dirname: str,
        trajectory_name: str,
        video_width: int = -1,
        video_height: int = -1,
        color_format: str = "rgb",
    ) -> Trajectory:
        trajectory = cls()
        trajectory.load_from_dir(
            dirname=dirname,
            trajectory_name=trajectory_name,
            video_width=video_width,
            video_height=video_height,
            color_format=color_format,
        )
        return trajectory

    def load_from_dir(
        self,
        dirname: str,
        trajectory_name: str,
        video_width: int = -1,
        video_height: int = -1,
        color_format: str = "rgb",
    ):
        self.color_format = color_format

        other_data_json_filename = Trajectory.get_data_path(dirname, trajectory_name)
        video_filename = Trajectory.get_video_path(dirname, trajectory_name)
        inputs_json_filename = Trajectory.get_user_inputs_path(dirname, trajectory_name)
        video_ticks_json_filename = Trajectory.get_video_ticks_path(
            dirname, trajectory_name
        )

        self.load(
            filename_by_name={
                Trajectory.VIDEO_KEY: video_filename,
                Trajectory.INPUTS_KEY: inputs_json_filename,
                Trajectory.VIDEO_TICKS_KEY: video_ticks_json_filename,
                Trajectory.DATA_KEY: other_data_json_filename,
            },
            video_width=video_width,
            video_height=video_height,
        )

    def load(
        self,
        filename_by_name: Dict[str, str],
        video_width: int = -1,
        video_height: int = -1,
    ):
        """
        Loads an Trajectory from optional path.json file (optionally with metrics) file, optional video filename,
        and optional user recording json file.  The video frames are loaded into a video_frames
        array and the images are scaled to the specified video width and height.
        """
        self.video_frames = []
        self.user_inputs = UserInputsLog()
        self.video_fps = None
        recorder_fps = -1

        for name, filename in filename_by_name.items():
            if not filename or not os.path.exists(filename):
                # Skip loading if the filename is empty or does not exist.
                continue
            if name == Trajectory.VIDEO_KEY:
                # While the VideoWriter takes BGR it creates a video in YUV format and the VideoReader
                # from the decord package returns the frames in RGB format.
                frames, recorder_fps = read_video(
                    filename, video_width, video_height, color_format="rgb"
                )
                self.video_frames = list(frames)
                self.video_source = filename
            elif name == Trajectory.INPUTS_KEY:
                self.user_inputs.load(filename)
            elif name == Trajectory.DATA_KEY:
                data = np.load(filename)
                self.other_data = {key: data[key] for key in data.files}
            elif name == Trajectory.VIDEO_TICKS_KEY:
                self.video_ticks = VideoTicks.load(filename)
                self.video_fps = int(self.video_ticks.video_fps)
                log.debug(
                    f"Video FPS (computed)={self.video_fps} and Recorder FPS = {recorder_fps}"
                )
            else:
                log.warning(
                    f"Unprocessed load name: {name}, only support {Trajectory.LOAD_SAVE_KEYS}"
                )

        if not self.video_fps:
            self.video_fps = recorder_fps

    def save_to_dir(
        self,
        dirname: str,
        trajectory_name: str,
        fourcc: str = "mp4v",
    ) -> List[str]:
        os.makedirs(dirname, exist_ok=True)

        other_data_json_filename = Trajectory.get_data_path(dirname, trajectory_name)
        video_filename = Trajectory.get_video_path(dirname, trajectory_name)
        inputs_json_filename = Trajectory.get_user_inputs_path(dirname, trajectory_name)
        video_ticks_json_filename = Trajectory.get_video_ticks_path(
            dirname, trajectory_name
        )

        return self.save(
            filename_by_name={
                Trajectory.VIDEO_KEY: video_filename,
                Trajectory.INPUTS_KEY: inputs_json_filename,
                Trajectory.VIDEO_TICKS_KEY: video_ticks_json_filename,
                Trajectory.DATA_KEY: other_data_json_filename,
            },
            fourcc=fourcc,
        )

    def save(self, filename_by_name: Dict[str, str], fourcc: str = "mp4v") -> List[str]:
        """
        Saves position and actions from trajectory to a .json file.
        Saves video frames to a .mp4 file.
        Saves user inputs (or predicted actions) to a .json file.
        :param filename_by_name: A dictionary mapping keys to filenames where the data should be saved.
        :param fourcc: The codec to use for saving the video file.
        :return: A list of filenames that were saved.
        """
        saved_files: List[str] = []
        for name, filename in filename_by_name.items():
            if not filename:
                # Skip saving if the filename is empty.
                continue
            if name == Trajectory.VIDEO_KEY:
                self._save_video(filename, fourcc_codec=fourcc)
                saved_files.append(filename)
            elif name == Trajectory.VIDEO_TICKS_KEY:
                self._save_video_ticks_json(filename)
                saved_files.append(filename)
            elif name == Trajectory.INPUTS_KEY:
                self._save_inputs_json(filename)
                saved_files.append(filename)
            elif name == Trajectory.DATA_KEY:
                self._save_other_data(filename)
                saved_files.append(filename)
            else:
                log.warning(
                    f"Unprocessed save name: {name}, only support {Trajectory.LOAD_SAVE_KEYS}"
                )
        return saved_files

    def _save_inputs_json(self, filename: str):
        if self.user_inputs:
            self.user_inputs.save(filename)

    def _save_video(self, filename: str, fourcc_codec: str = "mp4v"):
        if self.video_frames is not None and len(self.video_frames):
            save_video(
                filename,
                np.array(self.video_frames),
                self.video_fps,
                color_format=self.color_format,
                fourcc_codec=fourcc_codec,
            )

    def _save_video_ticks_json(self, filename: str):
        if self.video_ticks is not None:
            self.video_ticks.save(filename)

    def _save_other_data(self, file_name: str):
        if len(self.other_data):
            np.savez_compressed(file_name, **self.other_data)  # type: ignore
            log.info(f"Saved {len(self.other_data)} other_data in {file_name}")

    def add_step(
        self,
        frame: np.ndarray | None = None,
        video_tick: float | None = None,
        user_inputs: UserInputs | None = None,
        action_tick: float | None = None,
        other_data: dict = None,
    ):
        if frame is not None:
            self.video_frames.append(frame)
        if video_tick is not None:
            self.video_ticks.record(video_tick)
        if user_inputs:
            if action_tick:
                user_inputs.ticks = action_tick
            self.user_inputs.inputs += [user_inputs]
        if action_tick is not None:
            self.user_input_ticks += [action_tick]

        if other_data is not None:
            # Merge other_data into self.other_data so we collect a list
            # for each key in the dictionary.
            for key, value in other_data.items():
                if key not in self.other_data:
                    self.other_data[key] = []
                self.other_data[key].append(value)
