from safetensors import safe_open

import struct
import json
import io
import time
import os
import torch
from tqdm import tqdm
from torch.futures import Future
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, ReadItem

from typing import Any


def _get_safetensors_file_metadata(file_bytes: io.IOBase) -> tuple[Any, int]:
    # Copied from
    NUM_BYTES_FOR_HEADER_LEN = 8

    header_len_bytes = file_bytes.read(NUM_BYTES_FOR_HEADER_LEN)
    header_len = struct.unpack("<Q", header_len_bytes)[0]
    header_json = file_bytes.read(header_len)
    metadata = json.loads(header_json)
    return (metadata, header_len + NUM_BYTES_FOR_HEADER_LEN)


class SafetensorsReader(FileSystemReader):
    def __init__(self, path, sd):
        super().__init__(path, sd)
        self.sd = sd

    def read_data(self, plan: LoadPlan, planner: LoadPlanner):
        per_file: dict[str, list[ReadItem]] = {}
        for read_item in plan.items:
            item_md: _StorageInfo = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in tqdm(per_file.items()):
            new_path = self.fs.concat_path(self.path, relative_path)
            file_pointer = safe_open(new_path, framework="pt", device="cpu")
            for req in reqs:
                item_md = self.storage_data[req.storage_index]
                param = file_pointer.get_slice(req.storage_index.fqn)

                param = param[...]
                tensor = narrow_tensor_by_index(param, req.storage_offsets, req.lengths)
                target_tensor = planner.resolve_tensor(req).detach()

                assert target_tensor.size() == tensor.size(), (
                    f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                )
                target_tensor.copy_(tensor)
                planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def create_default_local_load_plan(
        state_dict: dict[str, Any], metadata: Metadata, strict: bool = True
    ) -> LoadPlan:
        return super().create_default_local_load_plan(
            state_dict=state_dict,
            metadata=metadata,
            strict=False,
        )

    def read_metadata(self) -> Metadata:
        meta = {}
        storage_data = {}
        for file in os.listdir(self.path):
            if file.endswith(".safetensors"):
                with self.fs.create_stream(
                    self.fs.concat_path(self.path, file), "rb"
                ) as f:
                    metadata, metadata_size = _get_safetensors_file_metadata(f)

                    for key, value in metadata.items():
                        if key == "__metadata__":
                            continue
                        if key not in meta:
                            md = TensorStorageMetadata(
                                properties=TensorProperties(dtype=torch.float32),
                                size=torch.Size(value["shape"]),
                                chunks=[
                                    ChunkStorageMetadata(
                                        offsets=torch.Size([0] * len(value["shape"])),
                                        sizes=torch.Size(value["shape"]),
                                    )
                                ],
                            )
                            meta[key] = md

                        else:
                            meta[key].chunks.append(
                                ChunkStorageMetadata(
                                    offsets=torch.Size([0] * len(value["shape"])),
                                    sizes=torch.Size(value["shape"]),
                                )
                            )

                        meta[key] = md
                        metadata_index = MetadataIndex(
                            fqn=key, offset=[0] * len(value["shape"])
                        )
                        storage_data[metadata_index] = _StorageInfo(
                            relative_path=file,
                            offset=value["data_offsets"][0] + metadata_size,
                            length=value["data_offsets"][1] - value["data_offsets"][0],
                        )

        metadata = Metadata(state_dict_metadata=meta, storage_data=storage_data)

        return metadata


def test_read():
    dist.init_process_group(
        "nccl", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"])
    )
    torch.cuda.set_device(f"cuda:{os.environ['RANK']}")

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear1 = nn.Linear(4, 4)
            self.linear2 = nn.Linear(8, 8)
            self.embed = nn.Embedding(16, 16)

    from torch.distributed.fsdp import fully_shard

    with torch.device("meta"):
        from transformers import AutoModelForCausalLM, AutoConfig

        config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
        model = AutoModelForCausalLM.from_config(config)

    model = fully_shard(model)

    model = model.to_empty(device="cuda")

    sd = model.state_dict()

    start_time = time.perf_counter()

    dcp.load(
        state_dict=sd,
        storage_reader=SafetensorsReader("out_large", None),
        planner=DefaultLoadPlanner(allow_partial_load=False),
    )

    end_time = time.perf_counter()
    print(f"Time taken to read safetensors: {end_time - start_time:.2f} seconds")

    sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()

    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    set_model_state_dict(model, sd)

    total_end_time = time.perf_counter()
    print(f"Total time taken: {total_end_time - start_time:.2f} seconds")
    return


if __name__ == "__main__":
    test_read()
