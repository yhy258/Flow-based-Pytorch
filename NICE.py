import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from flow_frame import *

class LogisticPrior(PriorFrame):
    def __init__(self):
        super().__init__()

    def log_prob(self, input):
        return -(F.softplus(input) + F.softplus(-input))

    def sample(self, shape):
        return torch.from_numpy(np.random.logistic(0, 1, shape)).float()


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, layer_num):
        super().__init__()
        self.in_module = nn.Sequential(
            nn.Linear(in_dim // 2, hid_dim),
            nn.ReLU(),
        )
        mid_module = []
        for _ in range(layer_num - 2):
            mid_module.append(nn.Linear(hid_dim, hid_dim))
            mid_module.append(nn.ReLU())

        self.mid_module = nn.Sequential(*mid_module)

        self.out_module = nn.Sequential(
            nn.Linear(hid_dim, in_dim // 2),
        )

    def forward(self, x):
        x = self.in_module(x)
        x = self.mid_module(x)
        return self.out_module(x)


class CouplingLayer(nn.Module):  # linear로 취급할거다 -> flatten 시킬거임
    def __init__(self, in_dim, hid_dim, layer_num, mask=False):
        super().__init__()
        self.mask = mask
        self.layer = MLP(in_dim, hid_dim, layer_num)

    def forward(self, input: Tensor, reverse=False):  # flatten 된 상태로 들어온다.

        BS, C = input.size()
        input = input.view(BS, C // 2, 2)

        if self.mask:
            input_1 = input[:, :, 0]  # [BS, C//2]
            input_2 = input[:, :, 1]
        else:
            input_1 = input[:, :, 1]
            input_2 = input[:, :, 0]

        shift = self.layer(input_1)

        if reverse == False:
            out_2 = input_2 + shift  # [BS, C//2]
        else:
            out_2 = input_2 - shift  # [BS, C//2]

        if self.mask:
            return torch.cat([input_1.unsqueeze(2), out_2.unsqueeze(2)], dim=2).view(BS, C)  # [BS, C]
        else:
            return torch.cat([out_2.unsqueeze(2), input_1.unsqueeze(2)], dim=2).view(BS, C)  # [BS, C]


class Scaling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros((1, dim)), requires_grad=True)

    def forward(self, x, p_log_det=None, reverse=False):
        if p_log_det == None:
            p_log_det = 0

        log_det = torch.sum(self.scale)

        if reverse == False:
            x = x * torch.exp(self.scale)
        else:
            x = x * torch.exp(-self.scale)

        return x, log_det + p_log_det


# Coupling Layer는 동일한 1 log det 내놓음. 신경 안써도 된다.
class NICE(FlowFrame):
    def __init__(self, coupling_num, in_dim, hid_dim, layer_num, prior_mode="logistic"):
        super().__init__()
        self.coupling_modules = nn.ModuleList(
            [CouplingLayer(in_dim, hid_dim, layer_num, mask=(i % 2)) for i in range(coupling_num)])
        self.scaling = Scaling(dim=in_dim)
        self.in_dim = in_dim

        assert prior_mode in ["logistic", "gaussian"], "prior mode는 logistic과 gaussian만 가능합니다."

        if prior_mode == "logistic":
            self.prior = LogisticPrior()
        else:
            self.prior = torch.distributions.Normal(torch.tensor(0.).to(device), torch.tensor(1.).to(device))

    def g(self, z: Tensor):  # z -> x
        x, _ = self.scaling(z, p_log_det=None, reverse=True)
        # coupling layer도 반대로 진행해줘야 한다.
        for layer in reversed(self.coupling_modules):
            x = layer(x, reverse=True)
        return x

    def f(self, x: Tensor):  # x -> z, log_prob
        z = x.view(x.size(0), -1)

        for layer in self.coupling_modules:
            z = layer(z)

        return self.scaling(z)

    def log_prob(self, x: Tensor):
        z, log_prob = self.f(x)
        ll_prob = torch.sum(self.prior.log_prob(z), dim=-1)
        return log_prob + ll_prob

    def sample(self, num, shape, device):
        z = self.prior.sample((num, self.in_dim)).to(device)
        return self.g(z).view(num, *shape)

    def forward(self, x):
        return self.log_prob(x)



if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NICE(coupling_num=4, in_dim=28 * 28, hid_dim=1000, layer_num=5).to(DEVICE)
    # model.load_state_dict(torch.load("/NICE"))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    dataset = datasets.MNIST(root="./data", train=True,
                             transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]),
                             download=True)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    EPOCHS = 50

    for epoch in range(1, EPOCHS + 1):
        print("{}/{} EPOCHS".format(epoch, EPOCHS))
        losses = []
        for img, _ in tqdm(train_loader):
            img = img.to(DEVICE)
            loss = -model(img).mean()
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Loss : {}".format(np.mean(losses)))

        torch.save(model.state_dict(), "/NICE")


    """
        Sampling
    """
    model.load_state_dict(torch.load("/NICE"))
    save_image(model.sample(10, (28, 28), device=DEVICE).view(10, 1, 28, 28), "/NICE2_logistic.png", nrow=5,
               normalize=False)
