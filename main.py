import dataclasses
import os
from io import UnsupportedOperation
from pathlib import Path
import pickle
from typing import cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import (
    _generate_uuid,
    _split_by_size_and_type,
)
from torch.distributed.checkpoint.metadata import MetadataIndex, StorageMeta
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint import SavePlan, SavePlanner, Metadata
from torch.distributed.checkpoint.storage import WriteResult
import queue

CHECKPOINT_DIR = "checkpoint"


class AccelerateStorageWriter(FileSystemWriter):
    _DEFAULT_SUFFIX = ".distcp"

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        self.optim_path = self.fs.concat_path(self.path, "optim")
        self.model_path = self.fs.concat_path(self.path, "model")
        self.fs.mkdir(self.optim_path)
        self.fs.mkdir(self.model_path)
        return super().prepare_local_plan(plan)

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ):
        storage_plan = plan.storage_data
        optim_file_count = 0
        model_file_count = 0

        def gen_file(is_optimizer: bool = False) -> str:
            nonlocal optim_file_count, model_file_count
            if is_optimizer:
                optim_file_count += 1
                return f"{storage_plan.prefix}{optim_file_count}{self._DEFAULT_SUFFIX}"
            else:
                model_file_count += 1
                return f"{storage_plan.prefix}{model_file_count}{self._DEFAULT_SUFFIX}"

        file_queue: queue.Queue = queue.Queue()

        for bucket in _split_by_size_and_type(1, plan.items):
            optim_states = [wi for wi in bucket if "optim" in wi.index.fqn]
            model_states = [wi for wi in bucket if "model" in wi.index.fqn]

            for state, path in zip(
                [optim_states, model_states], [self.optim_path, self.model_path]
            ):
                file_name = gen_file()
                path = self.fs.concat_path(path, file_name)
                file_queue.put((path, file_name, state))

        return self._write_data(planner, file_queue)

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        metadata = dataclasses.replace(metadata, version="1.0.0")

        def _split_metadata(
            metadata: Metadata,
        ) -> tuple[Metadata, Metadata]:
            result = []
            for to_get in ["model", "optim"]:
                result.append(
                    Metadata(
                        state_dict_metadata={
                            k.removeprefix("app."): v
                            for k, v in metadata.state_dict_metadata.items()
                            if to_get in k
                        },
                        planner_data={
                            k.removeprefix("app."): tuple([x for x in v if x != "app"])
                            for k, v in metadata.planner_data.items()
                            if to_get in k
                        },
                    )
                )

            return tuple(result)

        model_metadata, optim_metadata = _split_metadata(metadata)
        model_storage_md, optim_storage_md = {}, {}
        for wr_list in results:
            for wr in wr_list:
                new_index = dataclasses.asdict(wr.index)
                new_index["fqn"] = new_index["fqn"].removeprefix("app.")
                wr = WriteResult(
                    index=MetadataIndex(**new_index),
                    size_in_bytes=wr.size_in_bytes,
                    storage_data=wr.storage_data,
                )
                if "optim" in wr.index.fqn:
                    optim_storage_md.update({wr.index: wr.storage_data})
                else:
                    model_storage_md.update({wr.index: wr.storage_data})

        model_metadata.storage_data = model_storage_md
        optim_metadata.storage_data = optim_storage_md

        model_metadata.storage_meta = StorageMeta(
            self.checkpoint_id / Path("model"), save_id=_generate_uuid()
        )
        optim_metadata.storage_meta = StorageMeta(
            self.checkpoint_id / Path("optim"), save_id=_generate_uuid()
        )

        tmp_optim_path = cast(
            Path, self.fs.concat_path(self.optim_path, ".metadata.tmp")
        )
        tmp_model_path = cast(
            Path, self.fs.concat_path(self.model_path, ".metadata.tmp")
        )

        for meta, tmp_path, final_path in zip(
            [model_metadata, optim_metadata],
            [tmp_model_path, tmp_optim_path],
            [self.model_path, self.optim_path],
        ):
            with self.fs.create_stream(tmp_path, "wb") as metadata_file:
                pickle.dump(meta, metadata_file)
                if self.sync_files:
                    try:
                        os.fsync(metadata_file.fileno())
                    except (AttributeError, UnsupportedOperation):
                        os.sync()

            metadata_path = self.fs.concat_path(final_path, ".metadata")
            if self.fs.exists(metadata_path):
                self.fs.rm_file(metadata_path)

            self.fs.rename(tmp_path, metadata_path)


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_save_example(rank, world_size):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")
    setup(rank, world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = fully_shard(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    checkpoint_future = None
    for step in range(10):
        optimizer.zero_grad()
        model(torch.rand(8, 16, device="cuda")).sum().backward()
        optimizer.step()

        # waits for checkpointing to finish if one exists, avoiding queuing more then one checkpoint request at a time
        if checkpoint_future is not None:
            checkpoint_future.result()

        state_dict = {"app": AppState(model, optimizer)}
        checkpoint_future = dcp.async_save(
            state_dict,
            storage_writer=AccelerateStorageWriter(f"{CHECKPOINT_DIR}_step{step}"),
        )

    checkpoint_future.result()
    print(f"Checkpoint saved for rank {rank} at step {step}.")

    dcp.load(
        state_dict={
            "model": get_model_state_dict(model),
        },
        checkpoint_id=f"{CHECKPOINT_DIR}_step9/model",
    )
    dcp.load(
        state_dict={
            "optim": get_optimizer_state_dict(model, optimizer),
        },
        checkpoint_id=f"{CHECKPOINT_DIR}_step9/optim",
    )

    cleanup()


if __name__ == "__main__":
    world_size = 2
    print(f"Running async checkpoint example on {world_size} devices.")
    mp.spawn(
        run_fsdp_checkpoint_save_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
