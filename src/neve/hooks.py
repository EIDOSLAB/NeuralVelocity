import copy

import torch


def cosine_similarity(x1, x2, dim, eps=1e-8):
    x1_squared_norm = torch.pow(x1, 2).sum(dim=dim, keepdim=True)
    x2_squared_norm = torch.pow(x2, 2).sum(dim=dim, keepdim=True)

    x1_norm = x1_squared_norm.sqrt_()
    x2_norm = x2_squared_norm.sqrt_()

    x1_normalized = x1.div(x1_norm).nan_to_num(nan=0, posinf=0, neginf=0)
    x2_normalized = x2.div(x2_norm).nan_to_num(nan=0, posinf=0, neginf=0)

    mask_1 = (torch.abs(x1_normalized).sum(dim=dim) <= eps) * (torch.abs(x2_normalized).sum(dim=dim) <= eps)
    mask_2 = (torch.abs(x1_normalized).sum(dim=dim) > eps) * (torch.abs(x2_normalized).sum(dim=dim) > eps)

    cos_sim_value = torch.sum(x1_normalized * x2_normalized, dim=dim)

    return mask_2 * cos_sim_value + mask_1


class Hook:

    def __init__(self, name, module, momentum=0, use_gpu=False):
        self.name = name
        self.module = module

        self.activations = []
        self.previous_activations = None

        self.rho = 0
        self.velocity = 0

        self.use_gpu = use_gpu

        self.momentum = momentum

        self.active = True

        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if not self.active:
            return
        output_tensor = output.detach()
        if not self.use_gpu:
            output_tensor = output_tensor.cpu()
        current_activations = torch.movedim(output_tensor, 1, 0)
        current_activations = current_activations.reshape((current_activations.shape[0], -1))
        self.activations.append(current_activations)

    def _update_rho(self):
        similarity = cosine_similarity(
            self._get_current_activation().float(),
            self.previous_activations.float(),
            dim=1
        )
        self.rho = torch.clamp(similarity, -1., 1.)

    def _get_current_activation(self):
        return torch.cat(self.activations, dim=1)

    def get_velocity(self):
        self._update_rho()
        self.velocity = 1 - (self.rho + (self.momentum * self.velocity))
        return self.velocity.detach().cpu()

    def reset(self):
        current_activations = self._get_current_activation().detach()
        if not self.use_gpu:
            current_activations = current_activations.cpu()
        self.previous_activations = copy.deepcopy(current_activations)
        self.activations = []
        self.rho = 0

    def close(self) -> None:
        self.hook.remove()

    def activate(self, active):
        self.active = active
