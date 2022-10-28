# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from importlib import import_module
from typing import Dict, Tuple, Any

from functools import partialmethod
import sys
import logging
import timeit
import torch
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cli import LightningCLI


class Timer(object):
    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start


def instantiate_class(init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.
    Args:
        todo
    Returns:
        The instantiated class object.
    """
    kwargs = {k: init[k] for k in set(list(init.keys())) - {"_target_"}}

    class_module, class_name = init["_target_"].rsplit(".", 1)
    module = import_module(class_module, package=class_name)
    args_class = getattr(module, class_name)
    return args_class(**kwargs)


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def bootstrap(x: torch.Tensor, Nboot: int, binsize: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bootstrapping the mean of tensor

    Args:
        x (torch.Tensor): _description_
        Nboot (int): _description_
        binsize (int): _description_

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: bootstrapped mean and bootstrapped variance
    """
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    for i in range(Nboot):
        boots.append(torch.mean(x[torch.randint(len(x), (len(x),))], axis=(0, 1)))
    return torch.tensor(boots).mean(), torch.tensor(boots).std()


# From https://stackoverflow.com/questions/38911146/python-equivalent-of-functools-partial-for-a-class-constructor
def partialclass(name, cls, *args, **kwds):
    new_cls = type(name, (cls,), {"__init__": partialmethod(cls.__init__, *args, **kwds)})

    # The following is copied nearly ad verbatim from `namedtuple's` source.
    """
    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in enviroments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).
    """
    try:
        new_cls.__module__ = sys._getframe(1).f_globals.get("__name__", "__main__")
    except (AttributeError, ValueError):
        pass

    return new_cls


# CPU device metrics
_CPU_VM_PERCENT = "cpu_vm_percent"
_CPU_PERCENT = "cpu_percent"
_CPU_SWAP_PERCENT = "cpu_swap_percent"


def get_cpu_stats():
    import psutil

    return {
        _CPU_VM_PERCENT: psutil.virtual_memory().percent,
        _CPU_PERCENT: psutil.cpu_percent(),
        _CPU_SWAP_PERCENT: psutil.swap_memory().percent,
    }


class PDECLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("data.time_gap", "model.time_gap")
        parser.link_arguments("data.time_history", "model.time_history")
        parser.link_arguments("data.time_future", "model.time_future")
        parser.link_arguments("data.pde", "model.pdeconfig")
        parser.link_arguments("data.usegrid", "model.usegrid")
