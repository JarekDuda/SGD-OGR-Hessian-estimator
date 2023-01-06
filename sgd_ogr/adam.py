import math
from typing import List

import torch
from torch import Tensor
from torch.optim import Optimizer


class SimplifiedTorchAdam(Optimizer):
    """
    based on https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr,  betas=betas, eps=eps)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = torch.tensor(0.)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    state_steps.append(state['step'])

            adam(params_with_grad,
                 grads,
                 exp_avgs,
                 exp_avg_sqs,
                 state_steps,
                 beta1=beta1,
                 beta2=beta2,
                 lr=group['lr'],
                 eps=group['eps']
                 )
        return loss


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         state_steps: List[Tensor],
         *,
         beta1: float,
         beta2: float,
         lr: float,
         eps: float):

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m^t = \beta_1 * m^{t-1} + (1-\beta_1)*grad^t

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)  # v^t = \beta_2*v^{t-1}+(1-\beta_2)* (grad^t)**2

        step = step_t.item()

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr / bias_correction1  # \eta / (1-beta_1 ** t)

        bias_correction2_sqrt = math.sqrt(bias_correction2)  # sqrt(1-beta_2 ** t)

        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)  # sqrt(v^t) / sqrt(1-beta_2 ** t) + \eps

        param.addcdiv_(exp_avg, denom, value=-step_size)  # m^t / $denom * - (\eta / (1-beta_1 ** t))

