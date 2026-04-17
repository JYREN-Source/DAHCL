# losses/grl.py
import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):

        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, alpha):
        return GradientReversalFunction.apply(x, alpha)