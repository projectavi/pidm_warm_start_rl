# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from typing import NamedTuple, Tuple

import cv2
import numpy as np
import yaml

from pidm_imitation.constants import (
    DEFAULT_DATASET_FOLDER_NAME,
    OBSERVATIONS_FILE_SUFFIX,
    REPO_DIRECTORY_FOLDER_NAME,
    STATES_FILE_SUFFIX,
)
from pidm_imitation.utils.logger import Logger

log = Logger.get_logger(__name__)


class FileMetadata(NamedTuple):
    file_name: str
    sub_folder: str


def read_json(file: str) -> dict:
    with open(file, "r") as f:
        file_dict = json.load(f)
    return file_dict


def read_video(
    filepath: str, width: int = -1, height: int = -1, color_format: str = "rgb"
) -> Tuple[np.ndarray, int]:
    """
    Read video in mp4 file.
    :param path: The path to the file.
    :param width: Desired output width of the video, unchanged if `-1` is specified.
    :param height: Desired output height of the video, unchanged if `-1` is specified.
    :param color_format: The color format to use ("bgr" or "rgb").
    :return: The tuple containing the loaded mp4 data as numpy array and the video fps.
    """
    # Note: if we move this to the top of the file then pytest crashes on Linux.
    # So keep it here for now, thanks [chris]
    from decord import VideoReader

    video_reader = VideoReader(filepath, width=width, height=height)
    frames = video_reader[:].asnumpy()
    if color_format == "bgr":
        frames = frames[:, :, :, ::-1]
    return frames, int(video_reader.get_avg_fps())


def save_video(
    filepath: str,
    frames: np.ndarray,
    fps: int,
    width: int = -1,
    height: int = -1,
    color_format: str = "rgb",
    fourcc_codec: str = "mp4v",
) -> str:
    """
    Save frames in mp4 file.
    :param filepath: The path to the file.
    :param frames: The frames to save in the video.
    :param fps: The frames per second of the video.
    :param width: Desired width of the video, use frame dimensions if `-1` is specified.
    :param width: Desired height of the video, use frame dimensions if `-1` is specified.
    :param color_format: The color format of the frames ("bgr" or "rgb").
    :return: The path to the saved file.
    """
    assert len(frames) > 0, "No frames to save"
    assert color_format in ["bgr", "rgb"], f"Invalid color format: {color_format}"
    assert (
        frames.ndim == 4 and frames.shape[-1] == 3
    ), "Frames must be 3D arrays with shape (h, w, 3)"
    assert frames.dtype == np.uint8, "Frames must be of type uint8"
    width = width if width != -1 else frames[0].shape[1]
    height = height if height != -1 else frames[0].shape[0]

    fourcc: int = cv2.VideoWriter_fourcc(*fourcc_codec)  # type: ignore
    video_recorder = cv2.VideoWriter(
        filepath, fourcc, fps, (width, height), isColor=True
    )
    for im in frames:
        if color_format == "rgb":
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)  # video writer requires bgr format
        video_recorder.write(im)
    video_recorder.release()

    return filepath


def read_yaml(file: str) -> dict:
    with open(file, "r") as stream:
        file_dict = yaml.safe_load(stream)
    return file_dict


def save_yaml(data: dict, file: str) -> str:
    with open(file, "w") as stream:
        yaml.safe_dump(data, stream, default_flow_style=False)
    return file


def is_same_file(path1: str, path2: str):
    return os.path.realpath(path1) == os.path.realpath(path2)


def resolve_default_dataset_folder(data_dir: str) -> str:
    """Resolve the default dataset folder if the data_dir starts with the
    DEFAULT_DATASET_FOLDER_NAME environment variable.
    """
    if not data_dir:
        return None
    if data_dir.startswith(f"${DEFAULT_DATASET_FOLDER_NAME}"):
        dataset_folder = os.getenv(DEFAULT_DATASET_FOLDER_NAME)
        data_dir = data_dir.replace(f"${DEFAULT_DATASET_FOLDER_NAME}", dataset_folder)
    return data_dir


def resolve_repo_directory_folder(data_dir: str) -> str:
    """Resolve the repository directory folder if the data_dir starts with the
    REPO_DIRECTORY_FOLDER_NAME environment variable.
    """
    if not data_dir:
        return None
    if data_dir.startswith(f"${REPO_DIRECTORY_FOLDER_NAME}"):
        repo_folder = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        data_dir = data_dir.replace(f"${REPO_DIRECTORY_FOLDER_NAME}", repo_folder)
    return data_dir


def resolve_path(base_file: str, relative_path: str) -> str:
    """Extracts the directory name from the base file name and adds the relative path, for example
    if the base file is "/user/test/documents/file.txt" and the relative path is "../videos/test.mp4",
    then the result will be "/user/test/videos/test.mp4".
    """
    if not relative_path:
        return relative_path
    if not os.path.isabs(relative_path):
        relative_path = os.path.join(os.path.dirname(base_file), relative_path)
    return os.path.realpath(relative_path)


def list_files(path: str):
    """Generator that yields all files and their subfolders relative to the path, recursively"""
    for dirpath, dirnames, filenames in os.walk(path):
        if "wandb" in dirpath.split(os.path.sep):
            continue
        for file in filenames:
            file = os.path.join(dirpath, file)
            file_path, _ = os.path.split(file)
            sub_folder = os.path.relpath(file_path, path)
            if sub_folder == ".":
                sub_folder = ""
            yield sub_folder, file


def extract_file_name_from_path(file_name: str) -> str:
    """
    Given a file path, extract only the base name of the file, without
    the folder path and without its extension.
    """
    base_name = os.path.basename(file_name)
    # remove file extension
    base_name = os.path.splitext(base_name)[0]
    return base_name


def load_state_file(file: str, state_key: str) -> np.ndarray:
    """
    Load numpy arrays from file and return the array associated with the given key
    Args:
        file (str): The file path
        state_key (str): The key of the numpy array to return
    """
    data = np.load(file)
    # Display the length of the numpy arrays
    for key in data.files:
        log.info(f"Length of {key}: {len(data[key])}")
    return data[state_key]


def get_trajectory_prefix_from_state_filename(filename: str) -> str:
    """
    Extract trajectory prefix from a state filename by automatically detecting the state type.

    Args:
        filename: The filename (with or without .npz extension)

    Returns:
        The trajectory prefix (name without state suffix)
    """
    base_filename = filename[:-4] if filename.endswith(".npz") else filename

    for suffix in [OBSERVATIONS_FILE_SUFFIX, STATES_FILE_SUFFIX]:
        if suffix in base_filename:
            return base_filename.split(suffix)[0]

    raise ValueError(f"No valid state suffix found in filename '{filename}'")
