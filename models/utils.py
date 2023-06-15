import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, node_group=None):
        ctx.save_for_backward(input)
        ctx.node_group = node_group
        if node_group is None:
            output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
            dist.all_gather(output, input)
        else:
            output = [torch.zeros_like(input) for _ in range(dist.get_world_size(group=node_group))]
            dist.all_gather(output, input, group=node_group)
        return tuple(output)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, *grads):
        # all reduce grads
        coalesced = _flatten_dense_tensors(grads)
        if ctx.node_group is None:
            dist.all_reduce(coalesced)
        else:
            dist.all_reduce(coalesced, group=ctx.node_group)
        for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)

        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        if ctx.node_group is None:
            grad_out[:] = grads[dist.get_rank()]
        else:
            grad_out[:] = grads[dist.get_rank(group=ctx.node_group)]
        return grad_out, None
