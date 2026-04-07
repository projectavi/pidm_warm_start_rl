# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import random
from typing import Any, Dict, List, Tuple

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from pidm_imitation.agents.supervised_learning.dataset.align_dataset.alignment_strategy import (
    ActionFrameAlignmentStrategy,
)
from pidm_imitation.agents.supervised_learning.dataset.slicer import (
    check_slices,
    compute_slices,
)
from pidm_imitation.agents.supervised_learning.dataset.sliding_window_dataset import (
    SlidingWindowDataset,
)
from pidm_imitation.agents.supervised_learning.dataset.state_action_trajectory import (
    StateActionTrajectory,
    StateActionTrajectoryFactory,
)
from pidm_imitation.constants import INPUT_TRAJECTORIES_FILE, MODEL_DIRECTORY
from pidm_imitation.utils import Logger, StateType
from pidm_imitation.utils.padding_utils import ValidPadding
from pidm_imitation.utils.parsed_trajectory import ParsedTrajectory

logger = Logger()
log = logger.get_root_logger()


class DataModule(pl.LightningDataModule):
    TRAIN_TO_VAL_DATA_RATIO = 0.8

    def __init__(
        self,
        state_type: str,
        action_type: str,
        train_data_dir: str,
        history: int,
        history_slice_specs: Dict[str, Any] = {},
        lookahead: int = 0,
        lookahead_slice_specs: Dict[str, Any] = {},
        include_k: bool = False,
        padding_strategy: str = ValidPadding.NO_PADDING,
        align_strategy: ActionFrameAlignmentStrategy | None = None,
        validation_data_dir: str | None = None,
        batch_size: int = 32,
        num_workers: int = 32,
        num_train_samples: int | None = None,
        num_validation_samples: int | None = None,
        shuffle_trajectories: bool = True,
        shuffle_seed: int | None = None,
        output_dir: str = MODEL_DIRECTORY,
    ):
        """
        Data module for offline data. Load a train and validation datasets and create a dataloader for each.

        :param state_type: Type of the state used in the dataset.
        :param action_type: Type of the action used in the dataset.
        :param train_data_dir: directory where the train parsed trajectory data are stored
        :param history: Number of past frames to consider as history.
        :param history_slice_specs: Slice specifications for the history.
        :param padding_strategy: Padding strategy to use for the dataset.
        :param lookahead: Number of future frames to consider as lookahead.
        :param lookahead_slice_specs: Slice specifications for the lookahead.
        :param include_k: Whether to include the lookahead k in the data batch.
        :param align_strategy: Strategy(algorithm) to align/sync actions to frames in the dataset
        :param validation_data_dir: directory where the validation parsed trajectory data are stored
        :param batch_size: batch size for the dataloaders
        :param num_workers: number of workers per process
        :param num_train_samples: number of samples to use for training.
        :param num_validation_samples: number of samples to use for validation.
        :param shuffle_trajectories: whether to shuffle trajectories.
        :param shuffle_seed: seed to shuffle trajectories by. If not set, trajectories are not shuffled.
        :param output_dir: Directory where the model and other files will be saved.
        """
        super().__init__()
        self.save_hyperparameters()
        self.state_type = StateType.get_state_type_from_str(state_type)
        self.action_type = action_type
        self.train_data_dir = train_data_dir
        self.history = history
        self.history_slice_specs = history_slice_specs
        self.lookahead = lookahead
        self.lookahead_slice_specs = lookahead_slice_specs
        self.include_k = include_k
        self.padding_strategy = padding_strategy
        self.align_strategy = align_strategy
        self.validation_data_dir = validation_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train_samples = num_train_samples
        self.num_validation_samples = num_validation_samples
        self.shuffle_trajectories = shuffle_trajectories
        self.shuffle_seed = shuffle_seed
        self.output_dir = output_dir

        # parse dataset spec and extract parameters
        self.history_slice, self.num_history_samples = self._set_history_parameters()
        self.lookahead_slice, self.num_lookahead_samples = (
            self._set_lookahead_parameters()
        )
        assert (
            padding_strategy in ValidPadding.ALL
        ), f"padding_strategy must be one of {ValidPadding.ALL} but got {padding_strategy}"
        self.padding_strategy = padding_strategy

        # add extracted parameters to hparams. These are used to infer model parameters and link arguments.
        self.hparams["history_slice"] = self.history_slice
        log.info(f"DataModule history {self.history}")
        log.info(f"DataModule history slices: {self.history_slice}")
        if self.lookahead:
            self.hparams["lookahead_slice"] = self.lookahead_slice
            log.info(f"DataModule lookahead: {self.lookahead}")
            log.info(f"DataModule lookahead slices: {self.lookahead_slice}")

        # Initialize as None - will be populated in setup() or when first accessed
        self.train_trajs: None | List[ParsedTrajectory] = None
        self.validation_trajs: None | List[ParsedTrajectory] = None
        self._train_dataset: SlidingWindowDataset = None
        self._validation_dataset: SlidingWindowDataset = None
        self.output_files: List[str] | None = None

    def _initialize_data(self):
        """Initialize data loading and dataset creation. Can be called multiple times safely."""
        if self.train_trajs is None:
            # load data
            self.train_trajs, self.validation_trajs = self._load_data()

            # If any other file needs to be saved, add its absolute path to the list below
            self.output_files = self._save_datamodule_files()

            # Build dataset to compute the above so they can be linked to the model in the arguments in the CLI.
            self._build_dataset()

    def _set_history_parameters(self) -> Tuple[List[int], int]:
        """
        Extract and check history information in dataset args.
        :return: Tuple of (history_slice, num_history_samples)
        """
        if self.history > 0:
            history_slice = compute_slices(self.history, self.history_slice_specs)
            check_slices("history", self.history, history_slice)
        else:
            history_slice = []
        num_history_samples = len(history_slice)  # used by link_arguments

        return history_slice, num_history_samples

    def _set_lookahead_parameters(self) -> Tuple[List[int], int]:
        """
        Extract and check lookahead information in dataset args.
        :return: Tuple of (lookahead_slice, num_lookahead_samples)
        """
        if self.lookahead:
            lookahead_slice = compute_slices(self.lookahead, self.lookahead_slice_specs)
            check_slices("lookahead", self.lookahead, lookahead_slice)
        else:
            lookahead_slice = []
            self.include_k = False
        num_lookahead_samples: int = len(lookahead_slice)  # used by link_arguments
        return lookahead_slice, num_lookahead_samples

    def _save_datamodule_files(self) -> List[str]:
        """
        Save the train and validation trajectories names to a file, and (if used) the action bin strategy file.
        :return: List of absolute paths to all files that were saved by the datamodule.
        """
        saved_files = []
        os.makedirs(self.output_dir, exist_ok=True)
        input_path = os.path.abspath(
            os.path.join(self.output_dir, INPUT_TRAJECTORIES_FILE)
        )
        with open(input_path, "w") as f:
            traj_data = {
                "train": [traj.trajectory_name for traj in self.train_trajs],
                "validation": [traj.trajectory_name for traj in self.validation_trajs],
            }
            json.dump(traj_data, f, indent=4)
        saved_files.append(input_path)

        return saved_files

    def _check_if_any_samples(self, num_samples: int | None):
        """We use samples whenever num_samples is None (use all) or greater than 0
        for num_samples = 0, we don't use samples."""
        return num_samples is None or num_samples > 0

    def _validate_num_samples(self, dataset_len: int, num_samples: int | None):
        """Validate the number of samples requested from a dataset. If num_samples is invalid, raise an error."""
        if num_samples is not None:
            if num_samples <= 0:
                raise ValueError(
                    f"Number of samples ({num_samples}) must be a positive integer."
                )
            if num_samples > dataset_len:
                raise ValueError(
                    f"Number of samples requested ({num_samples}) is greater than the dataset length ({dataset_len})."
                )

    def _sort_trajectories_by_name(
        self, trajs: List[ParsedTrajectory]
    ) -> List[ParsedTrajectory]:
        return sorted(trajs, key=lambda x: x.trajectory_name)

    def _shuffle_trajectories(
        self, trajs: List[ParsedTrajectory]
    ) -> List[ParsedTrajectory]:
        """Randomly shuffle ordering of trajectories. Shuffling is determined by seed"""
        assert (
            self.shuffle_seed is not None
        ), "shuffle_seed must be set for reproducible shuffling of trajectories."
        random.seed(self.shuffle_seed)
        random.shuffle(trajs)  # in-place shuffling
        return trajs

    def _get_first_trajectories(
        self, trajs: List[ParsedTrajectory], num_trajs: int | None = None
    ) -> List[ParsedTrajectory]:
        """Select first X entries as subset of the data (trajs) if num_trajs is valid."""
        if num_trajs:
            self._validate_num_samples(len(trajs), num_trajs)
            return trajs[:num_trajs]
        else:
            return trajs

    def _train_val_split(
        self,
        sorted_train_trajs: List[ParsedTrajectory],
    ) -> Tuple[List[ParsedTrajectory], List[ParsedTrajectory]]:
        """Split the training data into training and validation sets.

        Three cases are considered:
        1. Both num_train_samples and num_validation_samples are set: sample the total number (their sum)
            and split accordingly
        2. Only num_train_samples is set: compute num_validation_samples to have the expected TRAIN_TO_VAL_DATA_RATIO
            and sample the total number (their sum)
        3. Both are None: split the train data by a constant ratio

        If only num_validation_samples is set, an error is raised.
        """
        if self.num_validation_samples and not self.num_train_samples:
            self.num_train_samples = (
                len(sorted_train_trajs) - self.num_validation_samples
            )
            assert self.num_train_samples > 0, (
                f"num_validation_samples ({self.num_validation_samples}) is greater than the"
                + f"number of samples ({len(sorted_train_trajs)})."
            )
        elif not self.num_validation_samples and self.num_train_samples:
            self.num_validation_samples = round(
                self.num_train_samples * (1.0 - DataModule.TRAIN_TO_VAL_DATA_RATIO)
            )
            log.info(
                f"num_train_samples is set but num_validation_samples is not. \
                     We've computed num_validation_samples as {self.num_validation_samples} to have the \
                     expected train/val ratio of {DataModule.TRAIN_TO_VAL_DATA_RATIO}."
            )
        elif not self.num_validation_samples and not self.num_train_samples:
            # Both are None, split the data by a constant ratio
            self.num_train_samples = round(
                len(sorted_train_trajs) * DataModule.TRAIN_TO_VAL_DATA_RATIO
            )
            self.num_validation_samples = (
                len(sorted_train_trajs) - self.num_train_samples
            )

        # Validate the number of samples requested
        self._validate_num_samples(len(sorted_train_trajs), self.num_train_samples)
        self._validate_num_samples(len(sorted_train_trajs), self.num_validation_samples)
        num_total_trajectories = self.num_train_samples + self.num_validation_samples

        # split the data into training and validation sets
        self._validate_num_samples(len(sorted_train_trajs), num_total_trajectories)
        train_trajs = sorted_train_trajs[: self.num_train_samples]
        validation_trajs = sorted_train_trajs[
            self.num_train_samples : num_total_trajectories
        ]

        return train_trajs, validation_trajs

    def _get_trajectory_state_data(self, data_dir: str) -> StateActionTrajectory:
        """Load the training and validation data."""
        traj_data = StateActionTrajectoryFactory.get_trajectory_by_state_type(
            state_type=self.state_type,
            datapath=data_dir,
            action_type=self.action_type,
            align_strategy=self.align_strategy,
        )
        return traj_data

    def _load_data(self):
        """Load the training and validation data."""
        train_data = self._get_trajectory_state_data(self.train_data_dir)
        assert len(train_data.parsed_traj) > 0, "No training data found."
        assert self._check_if_any_samples(
            self.num_train_samples
        ), "No training samples requested."

        # first sort trajectories by name to ensure fixed ordering after data loading
        # this is important for reproducible results across hardware
        train_trajs = self._sort_trajectories_by_name(train_data.parsed_traj)
        if self.shuffle_trajectories:
            # shuffle trajectories if specified for diverse sub-sets of data for training and validation
            train_trajs = self._shuffle_trajectories(train_trajs)

        if self.validation_data_dir:
            assert self._check_if_any_samples(
                self.num_validation_samples
            ), "No validation samples requested but specified validation directory."
            # If a validation directory is provided, no splitting is performed --> use specified
            # number of samples for training and validation
            validation_data = self._get_trajectory_state_data(self.validation_data_dir)

            # sort and potentially shuffle separate validation data
            validation_trajs = self._sort_trajectories_by_name(
                validation_data.parsed_traj
            )
            if self.shuffle_trajectories:
                validation_trajs = self._shuffle_trajectories(validation_trajs)

            # get first trajectories for training and validation in numbers as specified
            train_trajs = self._get_first_trajectories(
                train_trajs, self.num_train_samples
            )
            validation_trajs = self._get_first_trajectories(
                validation_trajs, self.num_validation_samples
            )
        elif self._check_if_any_samples(self.num_validation_samples):
            # If no validation directory is provided, but validation samples are requested, split the training data
            # into training and validation sets and sample the specified number of samples for each
            # (first num_train_samples for training and then num_validation_samples for validation)
            train_trajs, validation_trajs = self._train_val_split(train_trajs)
        else:
            # If no validation directory is provided and no validation samples are requested, use no validation data
            train_trajs = self._get_first_trajectories(
                train_trajs, self.num_train_samples
            )
            validation_trajs = []

        return train_trajs, validation_trajs

    def _build_dataset(self):
        assert (
            self.train_trajs
        ), "No training trajectories found. Please check the data directory."
        self._train_dataset = SlidingWindowDataset(
            history=self.history,
            history_slice=self.history_slice,
            lookahead=self.lookahead,
            lookahead_slice=self.lookahead_slice,
            parsed_trajectories=self.train_trajs,
            padding_strategy=self.padding_strategy,
        )
        if self.validation_trajs:
            self._validation_dataset = SlidingWindowDataset(
                history=self.history,
                history_slice=self.history_slice,
                lookahead=self.lookahead,
                lookahead_slice=self.lookahead_slice,
                parsed_trajectories=self.validation_trajs,
                padding_strategy=self.padding_strategy,
            )
        else:
            self._validation_dataset = None

    def prepare_data(self):
        log.info("Preparing data...")
        self._initialize_data()  # Ensure data is on disk before setup

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            log.info(f"Setting up training data on rank {self.trainer.global_rank}...")
            self._initialize_data()

    def _build_dataloader_kwargs(self, shuffle: bool) -> Dict[str, Any]:
        # Keep a few pinned batches staged so GPU copies do not sit idle behind Python workers.
        dataloader_kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = 4
        return dataloader_kwargs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            **self._build_dataloader_kwargs(shuffle=True),
        )

    def val_dataloader(self):
        return (
            DataLoader(
                self.validation_dataset,
                **self._build_dataloader_kwargs(shuffle=False),
            )
            if self.validation_dataset
            else []
        )

    @property
    def train_dataset(self):
        self._initialize_data()  # Ensure data is loaded
        return self._train_dataset

    @property
    def validation_dataset(self):
        self._initialize_data()  # Ensure data is loaded
        return self._validation_dataset

    @property
    def state_dim(self):
        return self.train_dataset.state_dim

    @property
    def target_dim(self):
        return self.train_dataset.action_dim

    @property
    def action_dim(self):
        return self.train_dataset.action_dim

    @property
    def sequence_length(self):
        return self.train_dataset.sequence_length

    @property
    def train_datasize(self):
        self._initialize_data()  # Ensure data is loaded
        return len(self.train_trajs)

    @property
    def validation_datasize(self):
        self._initialize_data()  # Ensure data is loaded
        return len(self.validation_trajs)

    @property
    def train_samples(self):
        return len(self.train_dataset)

    @property
    def validation_samples(self):
        return len(self.validation_dataset) if self.validation_dataset else 0
