import os
import random

import numpy as np
import torch


def set_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


@torch.no_grad()
def topk_accuracy(outputs, labels, topk=1):
    outputs = torch.softmax(outputs, dim=1)
    _, preds = outputs.topk(topk, dim=1)
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds)).sum()
    return (correct / float(len(outputs))).cpu().item()
