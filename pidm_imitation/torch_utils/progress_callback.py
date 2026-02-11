# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List

from pidm_imitation.utils import Logger

log = Logger.get_logger(__name__)


class TrainingProgressWatcher:
    def __init__(self):
        self.progress = 0.0

    def on_progress_update(self, progress: float):
        self.progress = progress


class TrainingProgressMixin:
    def __init__(self) -> None:
        self._progress_watchers: List[TrainingProgressWatcher]

    def initialize_watchers(self):
        self._progress_watchers = []

    @property
    def progress_watchers(self) -> List[TrainingProgressWatcher]:
        return self._progress_watchers

    def register_progress_watcher(self, watcher: TrainingProgressWatcher):
        self._progress_watchers.append(watcher)
