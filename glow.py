from flow_frame import *
from glow_utils import *
from glow_modules import *

class Flow(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.actnorm = Actnorm(in_channels)
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.coupling = Coupling(in_channels, hidden_channels)

    def forward(self, x, log_det=None, reverse=False):
        if log_det == None:
            log_det = 0

        if reverse:
            x, log_det = self.coupling(x, log_det, reverse=True)
            x, log_det = self.invconv(x, log_det, reverse=True)
            x, log_det = self.actnorm(x, log_det, reverse=True)
        else:
            x, log_det = self.actnorm(x, log_det, reverse=False)
            x, log_det = self.invconv(x, log_det, reverse=False)
            x, log_det = self.coupling(x, log_det, reverse=False)

        return x, log_det

class Glow(FlowFrame):
    def __init__(self, in_channels, hidden_channels, K, L):
        super().__init__()
        # squeeze 해서 들어갈거라서 *4 해줄거다.
        self.L = L
        self.K = K
        self.split_modules = nn.ModuleList([])
        self.flow_modules = nn.ModuleList([])
        for _ in range(L - 1):
            in_channels *= 4
            for _ in range(K):
                self.flow_modules.append(Flow(in_channels, hidden_channels))
            self.split_modules.append(SplitModule(in_channels))
            in_channels //= 2

        self.last_flow_modules = nn.ModuleList([])
        in_channels *= 4
        for _ in range(K):
            self.last_flow_modules.append(Flow(in_channels, hidden_channels))

    def forward(self, x, logdet=None):
        if logdet == None:
            logdet = 0
        return self.log_prob(x, logdet)

    def log_prob(self, x, logdet=None):
        bs, c, h, w = x.size()
        z, logdet = self.f(x, logdet)
        BS, C, H, W = z.size()
        zeros = torch.zeros(C, H, W).to(x.device)
        ll_prob = gaussian_likelihood(zeros, zeros, z)
        logdet = ll_prob + logdet
        logdet = (-logdet) / (math.log(2.0) * c * h * w)
        return logdet

    def sample(self, num, shape, device):
        C, H, W = shape
        C = C * 2 ** (self.L - 1) * 4
        H = H // (2 ** self.L)
        W = W // (2 ** self.L)
        z = torch.randn((num, C, H, W)).to(device)
        return self.g(z)[0].view(num, *shape)

    def f(self, x, logdet):
        bs, c, h, w = x.size()
        iter = 0
        for flow_layer in self.flow_modules:
            if iter % self.K == 0:
                x = squeeze2x2(x, reverse=False)
            x, logdet = flow_layer(x, logdet, reverse=False)

            iter += 1
            if iter % self.K == 0:
                x, logdet = self.split_modules[iter // self.K - 1](x, logdet, reverse=False)

        x = squeeze2x2(x, reverse=False)
        for flow_layer in self.last_flow_modules:
            x, logdet = flow_layer(x, logdet, reverse=False)
        return x, logdet

    def g(self, z, logdet=None):
        if logdet == None:
            logdet = 0
        for flow_layer in reversed(self.last_flow_modules):
            z, logdet = flow_layer(z, logdet, reverse=True)

        z = squeeze2x2(z, reverse=True)

        iter = 0
        for flow_layer in reversed(self.flow_modules):
            if iter % self.K == 0:
                z, logdet = self.split_modules[-(iter // self.K + 1)](z, logdet, reverse=True)

            iter += 1
            z, logdet = flow_layer(z, logdet, reverse=True)

            if iter % self.K == 0:
                z = squeeze2x2(z, reverse=True)
        return z, logdet

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, Actnorm):
                m.inited = True

if __name__ == "__main__":
    """
        Cifar example
    """
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    from torch.utils.data import DataLoader
    import numpy as np
    from tqdm import tqdm

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = 3
    hidden_channels = 512
    L = 3
    K = 32
    glow = Glow(in_channels, hidden_channels, K, L).to(DEVICE)
    optimizer = torch.optim.Adam(params=glow.parameters(), lr=1e-3, betas=[0.9, 0.9999], eps=1e-8)

    n_bits = 8


    def preprocess(x):
        # Follows:
        # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

        x = x * 255  # undo ToTensor scaling to [0,1]

        n_bins = 2 ** n_bits
        if n_bits < 8:
            x = torch.floor(x / 2 ** (8 - n_bits))
        x = x / n_bins - 0.5
        return x


    transformations = []
    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    dataloder = DataLoader(dataset, batch_size=64, shuffle=True)


    # https://github.com/chaiyujin/glow-pytorch/blob/master/glow/learning_rate_schedule.py
    def noam_learning_rate_decay(init_lr, global_step, warmup_steps=4000, minimum=None):
        # Noam scheme from tensor2tensor:
        warmup_steps = float(warmup_steps)
        step = global_step + 1.
        lr = init_lr * warmup_steps ** 0.5 * np.minimum(
            step * warmup_steps ** -1.5, step ** -0.5)
        if minimum is not None and global_step > warmup_steps:
            if lr < minimum:
                lr = minimum
        return lr

    """
        Train
    """
    EPOCHS = 50
    globalstep = 0
    for epoch in range(1, EPOCHS + 1):
        print("{}/{} EPOCHS".format(epoch, EPOCHS))
        losses = []
        for img, _ in tqdm(dataloder):
            lr = noam_learning_rate_decay(init_lr=1e-3, global_step=globalstep, warmup_steps=4000, minimum=1e-4)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            img = img.to(DEVICE)
            loss = glow(img).mean()
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(glow.parameters(), 5.)
            torch.nn.utils.clip_grad_norm_(glow.parameters(), 100.)
            optimizer.step()
            globalstep += 1

        print("Loss : {} Global Step : {}".format(np.mean(losses), globalstep))
        torch.save(glow.state_dict(), "/Glow")

    """
        Sampling
    """
    def postprocess(x):
        x = torch.clamp(x, -0.5, 0.5)
        x += 0.5
        x = x * 2 ** n_bits
        return torch.clamp(x, 0, 255)


    glow.set_actnorm_init()
    glow.eval()
    glow.load_state_dict(torch.load("/Glow"))

    save_image(postprocess(glow.sample(25, (3, 32, 32), device=DEVICE)).view(25, 3, 32, 32) / 255.,
               "/glow.png", nrow=5, normalize=False)
