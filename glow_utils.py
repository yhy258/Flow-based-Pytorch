import math
import torch

def squeeze2x2(input, reverse=False):
    B, C, H, W = input.size()

    if reverse:
        x = input.view(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // 4, H * 2, W * 2)

    else:
        x = input.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * 2 * 2, H // 2, W // 2)

    return x


def gaussian_p(mean, logs, x):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = logs ** 2
    """
    c = math.log(2 * math.pi)
    return -0.5 * (logs * 2.0 + ((x - mean) ** 2) / torch.exp(logs * 2.0) + c)


def gaussian_likelihood(mean, logs, x):
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])


def gaussian_sample(mean, logs):
    z = torch.normal(mean, torch.exp(logs))

    return z
