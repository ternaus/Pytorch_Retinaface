import torch
from typing import Tuple


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
    print(f"remove prefix '{prefix}'")

    def helper(x):
        return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {helper(key): value for key, value in state_dict.items()}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print(f"Missing keys:{len(missing_keys)}")
    print(f"Unused checkpoint keys:{unused_pretrained_keys}")
    print(f"Used keys:{used_pretrained_keys}")

    if len(used_pretrained_keys) == 0:
        raise ValueError("Load NONE from pretrained checkpoint.")

    return True


def load_model(model, pretrained_path, load_to_cpu):
    print(f"Loading pretrained model from {pretrained_path}")
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def split_array(array_length: int, num_splits: int, split_id: int) -> Tuple[int, int]:
    """Split array into parts.
    Args:
        array_length:
        num_splits:
        split_id:
    Returns: start and end indices of the
    """
    if not 0 <= split_id < num_splits:
        raise ValueError(f"gpu_id should be 0 <= {split_id} < {num_splits}")
    if array_length % num_splits == 0:
        step = int(array_length / num_splits)
    else:
        step = int(array_length / num_splits) + 1

    return split_id * step, min((split_id + 1) * step, array_length)
