from typing import Optional, List
from torch import Tensor
import torch
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def nested_tensor_from_tensor(tensor1: Tensor):
    # TODO make this more general
    if tensor1[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor1])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor1)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor1.dtype
        device = tensor1.device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), device=device)
        for img, pad_img, m in zip(tensor1, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = 0
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)
