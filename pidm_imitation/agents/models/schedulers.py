# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any

from torch.optim import lr_scheduler, Optimizer
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from transformers.optimization import get_scheduler as hf_get_scheduler


def get_scheduler(name: str, optimizer: Optimizer, **kwargs: Any) -> lr_scheduler.LRScheduler:

    # Note: most schedulers in pytorch are per epoch which makes clunky to use for our models
    # The schedulers in transformers.optimization are step based.
    num_warmup_steps = kwargs.pop("num_warmup_steps", 0)
    num_training_steps = kwargs.pop("num_training_steps", None)
    if name.lower() in TYPE_TO_SCHEDULER_FUNCTION:
        assert num_training_steps is not None, "num_training_steps is required for hf schedulers"
        return hf_get_scheduler(
            name,
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=kwargs,
        )
    elif name.lower().startswith("torch.optim.lr_scheduler."):
        # e.g. torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        scheduler_class = getattr(lr_scheduler, name.split(".")[-1])
        return scheduler_class(optimizer, **kwargs)
    else:
        raise ValueError(
            f"Unsupported scheduler: {name}. This supports any learning rate scheduler from "
            + "torch.optim.lr_scheduler or transformers.optimization.get_scheduler"
        )
