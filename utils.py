import torch
import torch.distributed as dist

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)