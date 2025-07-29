import torch

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from torch.distributed.checkpoint import (
    HuggingFaceStorageWriter,
    HuggingFaceStorageReader,
)


MODEL_ID = "Qwen/Qwen3-0.6B"
PATH = "outputs/checkpoint"


def init_distributed():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    torch.cuda.set_device(dist.get_rank())

def get_storage_writer(state_dict, checkpoint_id):
    fqn_to_index_mapping = {}
    num_fqns_per_file = 30
    # the use of 30 is just a heuristic for now.
    # Once these fqns map to HF ones, we can use the fqn mapping
    # from the model.safetensors.index.json file
    for i, key in enumerate(state_dict.keys()):
        group_num = (i // num_fqns_per_file) + 1
        fqn_to_index_mapping[key] = group_num

    storage_writer = HuggingFaceStorageWriter(
        path=checkpoint_id,
        save_distributed=True,
        fqn_to_index_mapping=fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )
    return storage_writer


def parallelize_module(model, device_mesh) -> dict:
    plan = model.config.base_model_tp_plan
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
    )
    from torch.distributed.tensor import Replicate

    _global_mapping = {
        "colwise": ColwiseParallel(),
        "rowwise": RowwiseParallel(),
        "colwise_rep": ColwiseParallel(output_layouts=Replicate()),
        "rowwise_rep": RowwiseParallel(input_layouts=Replicate()),
    }

    class_plan = {f"model.{k}": _global_mapping[v] for k, v in plan.items()}

    model = parallelize_module(
        model, device_mesh=device_mesh["tp"], parallelize_plan=class_plan
    )
    return model


def destroy_distributed():
    dist.destroy_process_group()


def main():
    init_distributed()
    device_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (4, 2), mesh_dim_names=("fsdp", "tp")
    )

    with torch.device("meta"):
        config = AutoConfig.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_config(config)

        model = parallelize_module(model, device_mesh)
        model = fully_shard(model, mesh=device_mesh["fsdp"])

    model.to_empty(device="cuda")
    dcp.load(
        storage_reader=HuggingFaceStorageReader(path=f"hf://{MODEL_ID}"),
        state_dict=model.state_dict(),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    inputs = tokenizer.apply_chat_template(
        conversation, tokenize=True, return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(inputs)
    if dist.get_rank() == 0:
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
