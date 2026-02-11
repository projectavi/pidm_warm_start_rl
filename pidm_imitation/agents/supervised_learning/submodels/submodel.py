# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC

import lightning.pytorch as pl


class SubModel(pl.LightningModule, ABC):
    """
    Base class for submodels in the supervised learning framework.
    Mainly serves to provide utility functions of lightning such as freeze.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
