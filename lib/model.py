import torch
import torch.nn as nn
import re
import math
import torch.nn.functional as F
from typing import Callable


class ModelUtils:

    @staticmethod
    def get_act_func_by_name(act_func_name, inplace=False):
        """Get act functions by name."""
        if act_func_name == "SiLU":
            act_func = nn.SiLU(inplace=inplace)
        elif act_func_name == "ReLU":
            act_func = nn.ReLU(inplace=inplace)
        elif act_func_name.startswith("lReLU"):
            try:
                act_func = nn.LeakyReLU(
                    float(re.findall(r"[-+]?\d*\.\d+|\d+", act_func_name)[0]),
                    inplace=inplace,
                )
            except Exception:
                act_func = nn.LeakyReLU(0.1, inplace=inplace)
        elif act_func_name == "Sigmoid":
            act_func = nn.Sigmoid()
        elif act_func_name == "Softplus":
            act_func = nn.Softplus()
        else:
            raise TypeError("No such activation function")
        return act_func

    @staticmethod
    def get_time_embedding_func_by_name(
        embedding_func_name: str,
    ) -> Callable[[torch.Tensor, int], torch.Tensor]:
        
        """Get time embedding function by name.

        Including 'sin' or 'onehot'
        """
        match embedding_func_name:
            case "sin":
                embedding_func = ModelUtils.get_timestep_sin_embedding
            case "onehot":
                embedding_func = ModelUtils.get_timestep_onehot_embedding
            case _:
                raise NotImplementedError
        return embedding_func

    @staticmethod
    def get_timestep_sin_embedding(timesteps, embedding_dim):
        """This is different from the description in Section 3.5 of 'Attention
        Is All You Need'

        TE(pos, i) = sin(\\frac{pos}{10000^{i/(embedding_dim/2-1)}}),
            i form 0 ~ embedding/2-1
        And for i+embedding/2-1, change sin to cos

        if embedding_dim is odd, then zero padding.

        NOTE: The only difference between Attention and there is
        former is embedding/2, latter is embedding/2-1. if embedding_dim
        is 128, then former:
        TE(pos, i) = sin(\\frac{pos}{10000^{i/(64)}}),
            i form 0 ~ 63
        while latter changes 64 to 63. (Is there Any difference?)

        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * (-emb))
        emb = emb.to(timesteps.device)
        emb = timesteps.float() * emb[None, :]  # timesteps: [Batch_size, 1]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero padding
            emb = nn.functional.pad(emb, (0, 2, 0, 0))
        return emb  # [Batch_size, embedding_dim]

    @staticmethod
    def get_timestep_onehot_embedding(timesteps, embedding_dim=None):
        """Get one-hot time embeddings.

        If embedding_dim is not given, then num_class=max(timesteps)+1

        Args:
            timesteps: [B, 1]
            embedding_dim: int (default None)
        Returns:
            onehot_encoding: [B, embedding_dim]
        """

        if embedding_dim is None:
            embedding_dim = int(torch.max(timesteps) + 1)

        assert embedding_dim >= int(torch.max(timesteps) + 1)

        one_hot_encoding = F.one_hot(
            timesteps.squeeze(), num_classes=embedding_dim
        ).float()
        return one_hot_encoding  # [B, embedding_dim]


class MyUfunc(nn.Module):
    """The same as 'Gauss1D' in DAEBM paper.

    U_theta (x) = (x/10) ^ 2 + theta ^ T h_theta (x)
    where h_theta (x) = ( relu(x-ksai_1), relu(x-ksai_2), ..., relu(x-ksai_K))

    """

    def __init__(self):
        super().__init__()
        self.ch = 1
        self.ksai = torch.arange(-4, 4, 0.1)
        self.out_ch = self.ksai.shape[0]
        self.linear = nn.Linear(self.out_ch, self.ch, bias=False)

    def init_theta(self, mean=0, std=1):
        """initialize parameters of model, using normal"""

        torch.nn.init.normal_(self.linear.weight, mean=mean, std=std)

    def forward(self, x):
        # shape of x should be [B, 1]
        h = torch.abs(x - self.ksai)
        h = self.linear(h)
        energy = torch.pow(x / 10, 2) + h
        return energy


class MyUfuncOneD(nn.Module):
    """Using MLP as energy function."""

    def __init__(self, in_ch: int = 1, mid_ch: int = 8, act_func: str = "ReLU"):
        super().__init__()
        self.ch = in_ch
        self.mid_ch = mid_ch

        self.linear1 = nn.Linear(self.ch, self.mid_ch)
        self.linear2 = nn.Linear(self.mid_ch, self.mid_ch)
        self.linear3 = nn.Linear(self.mid_ch, 1)
        self.init_theta()

        self.act_func = ModelUtils.get_act_func_by_name(act_func_name=act_func)

    def init_theta(self, mean=0, std=1):
        """Initialize parameters of model

        Using normal
        """

        torch.nn.init.normal_(self.linear1.weight, mean=mean, std=std)
        torch.nn.init.constant_(self.linear1.bias, 0.0)

        torch.nn.init.normal_(self.linear2.weight, mean=mean, std=std)
        torch.nn.init.constant_(self.linear2.bias, 0.0)

        torch.nn.init.normal_(self.linear3.weight, mean=mean, std=std)
        torch.nn.init.constant_(self.linear3.bias, 0.0)

    def forward(self, x):
        x = self.act_func(self.linear1(x))
        x = self.act_func(self.linear2(x))
        x = self.linear3(x)
        return x


class MyUfuncTemb(nn.Module):
    """According to "ToyTembModel" in DAEBM.

    Network contains a Four-layer MLP, which has 128
    hidden neurons in each middle layers. Time embedding using
    sinusoidal time embedding or one-hot embedding

    """

    def __init__(self, in_ch, n_timesteps, act_func="SiLU", time_embedding_type="sin"):
        super().__init__()
        self.ch = 16
        self.temb_ch = 16
        self.n_timesteps = n_timesteps

        self.lin1 = nn.Linear(in_ch, self.ch)
        self.lin2 = nn.Linear(self.ch, self.ch)
        self.lin4 = nn.Linear(self.ch, 1)

        self.temb_proj1 = nn.Linear(self.temb_ch, self.ch)
        self.temb_proj2 = nn.Linear(self.temb_ch, self.ch)

        self.act_func = ModelUtils.get_act_func_by_name(act_func)
        self.time_embedding_func = ModelUtils.get_time_embedding_func_by_name(
            time_embedding_type
        )

    @property
    def n_class(self):
        return self.n_timesteps

    def forward(self, x, t):
        temb = self.time_embedding_func(t, self.temb_ch)
        x = self.act_func(self.lin1(x.flatten(1)) + self.temb_proj1(temb))
        x = self.act_func(self.lin2(x) + self.temb_proj2(temb))
        x = self.lin4(x)
        return x  # [Batch_size, 1]

    def f(self, x):
        """Generate U(x, t) for t from 0 ~ T-1

        Used in MGMS, we need to calculate p(t|x)
        """
        output = [
            self.forward(x, t.long().repeat(x.shape[0]).unsqueeze(-1).to(x.device))
            for t in torch.arange(self.n_timesteps)
        ]
        output = torch.stack(output, dim=1)
        return output  # [Batch_size, T, 1]

    def energy_output(self, x, t=None):
        """Get -U(x,t), t=0 if t is not given."""
        if t is None:
            t = torch.zeros((x.shape[0], 1)).long().to(x.device)
        energy = -self.forward(x, t)
        return energy
