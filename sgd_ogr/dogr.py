import math
from typing import List

import torch
from torch import Tensor
from torch.optim import Optimizer


class dOGR(Optimizer):

    def __init__(self, params, lr=0.5, beta=0.6, div=1.5, cut=5.0, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter value: {}".format(beta))

        defaults = dict(lr=lr, beta=beta, div=div, cut=cut, eps=eps)
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

            exp_avg_mts = []
            exp_avg_mgs = []
            exp_avg_ms = []

            exp_avg_sq_tts = []
            exp_avg_sq_gts = []

            state_steps = []

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

                        state['exp_avg_mt'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_mg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_m'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        state['exp_avg_sq_tt'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq_gt'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg_mts.append(state['exp_avg_mt'])
                    exp_avg_mgs.append(state['exp_avg_mg'])
                    exp_avg_ms.append(state['exp_avg_m'])

                    exp_avg_sq_tts.append(state['exp_avg_sq_tt'])
                    exp_avg_sq_gts.append(state['exp_avg_sq_gt'])

                    state_steps.append(state['step'])

            dogr(params_with_grad,
                 grads,
                 exp_avg_mts,
                 exp_avg_mgs,
                 exp_avg_ms,
                 exp_avg_sq_tts,
                 exp_avg_sq_gts,
                 state_steps,
                 beta=group['beta'],
                 lr=group['lr'],
                 div=group['div'],
                 cut=group['cut'],
                 eps=group['eps'],
                 )

        return loss


def dogr(params: List[Tensor],
         grads: List[Tensor],
         exp_avg_mts: List[Tensor],
         exp_avg_mgs: List[Tensor],
         exp_avg_ms: List[Tensor],
         exp_avg_sq_tts: List[Tensor],
         exp_avg_sq_gts: List[Tensor],
         state_steps: List[Tensor],
         *,
         beta: float,
         lr: float,
         div: float,
         cut: float,
         eps: float):
    for i, param in enumerate(params):
        grad = grads[i]

        exp_avg_mt = exp_avg_mts[i]
        exp_avg_mg = exp_avg_mgs[i]
        exp_avg_m = exp_avg_ms[i]

        exp_avg_sq_tt = exp_avg_sq_tts[i]
        exp_avg_sq_gt = exp_avg_sq_gts[i]

        step_t = state_steps[i]

        # update step
        step_t += 1

        exp_avg_mt.mul_(beta).add_(param, alpha=1 - beta)
        exp_avg_mg.mul_(beta).add_(grad, alpha=1 - beta)
        exp_avg_m.mul_(1.0 - lr).add_(grad, alpha=lr)

        dtm = (param - exp_avg_mt)
        dgm = (grad - exp_avg_mg)

        exp_avg_sq_tt.mul_(beta).addcmul_(dtm, dtm.conj(), value=1 - beta).add_(eps)
        exp_avg_sq_gt.mul_(beta).addcmul_(dgm, dtm.conj(), value=1 - beta)

        denom = torch.abs(exp_avg_sq_gt / exp_avg_sq_tt).mul_(div).clamp_max_(cut).mul_(exp_avg_m).add_(eps)
        param.add_(torch.div(-1, denom))
