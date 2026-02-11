# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

CONFIG_FILE_KEY = "config_path"

INPUTS_FILE_SUFFIX = "_inputs"
VIDEO_METADATA_SUFFIX = "_video_ticks"
VIDEO_FILE_SUFFIX = "_video"
STATES_FILE_SUFFIX = "_states"
OBSERVATIONS_FILE_SUFFIX = "_observations"
TRAJECTORY_DATA_FILE_SUFFIX = "_trajectory_data"
ENV_CONFIG_FILE_SUFFIX = "_env_config"

REPO_DIRECTORY_FOLDER_NAME = "REPO_DIRECTORY"
DEFAULT_DATASET_FOLDER_NAME = "DEFAULT_DATASET_FOLDER"
LAST_CHECKPOINT = "last"
MODEL_DIRECTORY = "checkpoints"
CHECKPOINT_EXTENSION = ".ckpt"

# Dataset constants
STATE_KEY = "state"
STATE_HISTORY_KEY = "state_history"
ACTION_HISTORY_KEY = "action_history"
ACTION_TARGET_KEY = "action_target"
LOOKAHEAD_K_KEY = "lookahead_k"
LOOKAHEAD_K_ONEHOT_KEY = "lookahead_k_onehot"
STATE_LOOKAHEAD_KEY = "state_lookahead"
ACTION_LOOKAHEAD_KEY = "action_lookahead"

DATASET_KEYS = [
    STATE_KEY,
    STATE_HISTORY_KEY,
    ACTION_HISTORY_KEY,
    ACTION_TARGET_KEY,
    LOOKAHEAD_K_KEY,
    LOOKAHEAD_K_ONEHOT_KEY,
    STATE_LOOKAHEAD_KEY,
    ACTION_LOOKAHEAD_KEY,
]


# Submodel names
POLICY_MODEL_KEY = "policy"
STATE_ENCODER_MODEL_KEY = "state_encoder"
POLICY_HEAD_KEY = "policy_head"

# Keys for different model outputs/ predictions
PREDICTED_ACTION_KEY = "predicted_action"

# General keys and suffixes
HISTORY_KEY = "history"
LOOKAHEAD_KEY = "lookahead"

# State encoder
STATE_ENCODER_INPUT_GROUP_KEYS = [
    HISTORY_KEY,
    LOOKAHEAD_KEY,
]
EMB_KEY_SUFFIX = "_emb"
HISTORY_EMB_KEY = HISTORY_KEY + EMB_KEY_SUFFIX
LOOKAHEAD_EMB_KEY = LOOKAHEAD_KEY + EMB_KEY_SUFFIX
ENCODER_EMB_KEY = "encoder" + EMB_KEY_SUFFIX

# Inference model input keys (must match the argument names on SlidingWindowInferenceIdmModel)
INFERENCE_STATE_KEY = "state"
INFERENCE_ACTION_KEY = "action"
INFERENCE_LOOKAHEAD_KEY = "lookahead"
INFERENCE_LOOKAHEAD_ACTION_KEY = "lookahead_action"

# Logging keys
TRAIN_PREFIX = "train"
VALIDATION_PREFIX = "validation"
LOSS_SUFFIX = "_total_loss"
TRAIN_LOSS = TRAIN_PREFIX + LOSS_SUFFIX
VALIDATION_LOSS = VALIDATION_PREFIX + LOSS_SUFFIX

OUTPUT_DIR = "output"
INPUT_DIR = "input"
INPUT_TRAJECTORIES_FILE = "input_trajectories.json"

# Controller constants
XBOX_ONE_CONTROLLER_NAME = "Controller (Xbox One For Windows)"
XBOX_X_CONTROLLER = "Xbox Series X Controller"
XBOX_ONE_S_CONTROLLER = "Xbox One S Controller"
XBOX_360_CONTROLLER = "Xbox 360 Controller"
XBOX_CONTROLLERS = [
    XBOX_ONE_CONTROLLER_NAME,
    XBOX_X_CONTROLLER,
    XBOX_ONE_S_CONTROLLER,
    XBOX_360_CONTROLLER,
]
