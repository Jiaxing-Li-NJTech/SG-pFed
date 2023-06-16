import torch
import numpy as np
import torch.nn.functional as F



def torch_tile(tensor, dim, n):
    if dim == 0:
        return tensor.unsqueeze(0).transpose(0, 1).repeat(1, n, 1).view(-1, tensor.shape[1])
    else:
        return tensor.unsqueeze(0).transpose(0, 1).repeat(1, 1, n).view(tensor.shape[0], -1)


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [bs, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [bs, num_classes, *]
    """
    result = torch.zeros(len(input),num_classes).cuda()
    result = result.scatter_(1, input.unsqueeze(1), 1)

    return result

def get_center_matrix(logits, labels,NUM_CLASSES):
    labels = make_one_hot(labels,NUM_CLASSES)
    source_prob = []

    for i in range(NUM_CLASSES):
        mask = torch_tile(torch.unsqueeze(labels[:, i], -1), 1, logits.shape[1])
        logits_mask_out = logits * mask
        logits_avg = torch.sum(logits_mask_out, dim=0) / (torch.sum(labels[:, i]) + 1e-8)
        source_prob.append(logits_avg)
    return torch.stack(source_prob)

