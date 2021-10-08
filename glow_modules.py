import torch
import torch.nn as nn
import torch.nn.functional as F

from glow_utils import *

class Actnorm(nn.Module):
    def __init__(self, feature_dim, scale=1.0, non_return_det=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.scale = scale
        self.non_return_det = non_return_det

        self.bias = nn.Parameter(torch.zeros(1, feature_dim, 1, 1))
        self.log = nn.Parameter(torch.zeros(1, feature_dim, 1, 1))

        self.inited = False

    @torch.no_grad()
    def param_initialize(self, input):
        bias = -torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
        var = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
        log = torch.log(self.scale / (torch.sqrt(var) + 1e-10))
        self.bias.data.copy_(bias.data)
        self.log.data.copy_(log.data)

        self.inited = True

    def forward(self, x, log_det=None, reverse=False):
        if log_det == None:
            log_det = 0
        bs, c, h, w = x.size()
        if self.inited == False:
            self.param_initialize(x)

        slogdet = torch.sum(self.log) * h * w
        if reverse:
            x = torch.exp(-self.log) * x - self.bias
            slogdet *= -1

        else:
            x = torch.exp(self.log) * (x + self.bias)

        log_det += slogdet
        if self.non_return_det:
            return x
        return x, log_det


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.linalg.qr(torch.randn(*w_shape), mode="complete")[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)

        if logdet == None:
            logdet = 0

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class SplitModule(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = ConvZero(num_channels // 2, num_channels, 3, 1, 1)

    def reverse_true(self, z, logdet):
        z1 = self.conv(z)
        mu, log = z1[:, 0::2, ...], z1[:, 1::2, ...]
        z2 = gaussian_sample(mu, log)
        z = torch.cat([z, z2], dim=1)
        return z, logdet

    def reverse_false(self, z, logdet):
        z1, z2 = torch.chunk(z, chunks=2, dim=1)
        h = self.conv(z1)
        mu, log = h[:, 0::2, ...], h[:, 1::2, ...]
        return z1, gaussian_likelihood(mu, log, z2) + logdet

    def forward(self, z, logdet, reverse=False):
        if logdet == None:
            logdet = 0
        if reverse:
            return self.reverse_true(z, logdet)
        else:
            return self.reverse_false(z, logdet)


class ConvZero(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, logscale=3.):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.logscale = logscale
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, x):
        x = self.conv(x)
        return x * torch.exp(self.logs * self.logscale)


class Coupling(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, mid_channel, 3, padding=1, stride=1),
            Actnorm(mid_channel, scale=1.0, non_return_det=True),
            nn.ReLU(),
            nn.Conv2d(mid_channel, mid_channel, 1, padding=0, stride=1),
            Actnorm(mid_channel, scale=1.0, non_return_det=True),
            nn.ReLU(),
            ConvZero(mid_channel, in_channel, 3, padding=1, stride=1),  # zero module로 바꿔주자.
        )

    def forward(self, x, log_det=None, reverse=False):
        # masking..
        if log_det == None:
            log_det = 0
        bs = x.size(0)
        z1, z2 = torch.chunk(x, chunks=2, dim=1)
        h = self.net(z1)
        s, t = h[:, 0::2, ...], h[:, 1::2, ...]
        s = torch.sigmoid(s + 2.0)

        if reverse:
            z2 = z2 / s - t
            log_det -= torch.sum(torch.log(s).view(bs, -1), dim=-1)

        else:
            z2 = s * (z2 + t)
            log_det = torch.sum(torch.log(s).view(bs, -1), dim=-1) + log_det

        out = torch.cat([z1, z2], dim=1)
        return out, log_det
