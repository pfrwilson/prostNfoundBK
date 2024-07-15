from argparse import ArgumentParser
from dataclasses import dataclass
import json
from shutil import rmtree
import torch
import os
from torch.distributed import init_process_group, barrier
import logging
import sys
from os.path import join
from logging import Logger
from wandb.wandb_run import Run
import wandb


@dataclass
class ExperimentUtility:
    exp_dir: str
    ckpt_dir: str
    logger: Logger
    rank: int = 0
    world_size: int = 1
    state: dict | None = None
    wandb_run: Run | None = None

    def log_info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def log_metrics(self, metrics: dict):
        if self.wandb_run is None:
            self.log_info(str(metrics))
        else:
            self.wandb_run.log(metrics)

    def save_checkpoint(self, state: dict, filename="state.pt"):
        torch.save(state, join(self.ckpt_dir, filename))
        self.logger.info(f"Saved checkpoint to {join(self.ckpt_dir, filename)}")


def setup(
    exp_dir,
    ckpt_dir_symlink_target=None,
    use_distributed: bool = False,
    debug: bool = False,
    conf: dict = {},
    ckpt_file="state.pt",
    use_wandb=False,
    wandb_kwargs: dict = {},
    overwrite: bool = False,
):
    # distributed setup

    if not torch.cuda.is_available():
        raise RuntimeError("Could not find gpu.")
    # First thing we do is check for distributed training
    if use_distributed:
        VALID_ENVIRONMENT = (
            "MASTER_ADDR" in os.environ,
            "MASTER_PORT" in os.environ,
            "RANK" in os.environ,
            "WORLD_SIZE" in os.environ,
        )
        assert (
            VALID_ENVIRONMENT
        ), f"We need to have the necessary environment variables to run multiprocessing."
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        rank = 0
        world_size = 1
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    # =========== BASIC LOGGING ================
    logger = logging.getLogger("Main")
    logger.propagate = False
    # format the logger
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if rank == 0:
        for handler in logger.handlers:
            logger.removeHandler(handler)
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        logger.setLevel(logging.CRITICAL)

    logger.info(f"Beginning setup.")
    logger.info(f"Using {world_size} processes.")
    logger.info(f"Using rank {rank}.")

    # ============ EXPERIMENT DIRECTORY ============
    logger.info("Setting up experiment directory.")

    if rank == 0:
        if os.path.exists(exp_dir):
            logger.info(f"Found experiment directory {exp_dir}.")
            if overwrite:
                logger.info("Overwriting!")
                rmtree(exp_dir)
                os.makedirs(exp_dir, exist_ok=True)
        else:
            logger.info(f"Creating new experiment directory.")
            os.makedirs(exp_dir, exist_ok=True)
        if ckpt_dir_symlink_target is not None:
            if not os.path.exists(join(exp_dir, "checkpoints")):
                os.symlink(
                    ckpt_dir_symlink_target,
                    join(exp_dir, "checkpoints"),
                    target_is_directory=True,
                )
        else:
            ckpt_dir_symlink_target = join(exp_dir, "checkpoints")
            if not os.path.exists(join(exp_dir, "checkpoints")):
                os.makedirs(join(exp_dir, "checkpoints"))
            logging.info(f"Created checkpoint directory {join(exp_dir, 'checkpoints')}")

        with open(join(exp_dir, "config.json"), "w") as f:
            json.dump(conf, f)

    if use_distributed:
        barrier()  # we need to make sure the other processes wait for the above to be done

    # =========== ADVANCED LOGGING ================
    # we also log to a file
    logger.info("Setting up file logging.")
    if rank == 0:
        file_handler = logging.FileHandler(join(exp_dir, "out.log"))
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # we also log to wandb
    if rank == 0 and use_wandb and not debug:
        logger.info("Setting up wandb logging.")
        wandb.init(config=conf, **wandb_kwargs)
    else:
        logger.info("Not setting up wandb logging.")

    # load state
    state_path = os.path.join(exp_dir, "checkpoints", ckpt_file)
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        logger.info(f"Found checkpoint with keys {state.keys()}!")
    else:
        state = None
        logger.info(f"No saved checkpoint state found")

    return ExperimentUtility(
        exp_dir,
        join(exp_dir, "checkpoints"),
        logger,
        rank,
        world_size,
        state,
        wandb.run,
    )

