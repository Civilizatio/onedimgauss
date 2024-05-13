import torch
from torch.utils.data import Dataset
from scipy.stats import norm
from typing import Literal


class _MyDataset(Dataset):
    """One dimesion Gaussian ( Can be substituded by GaussFourDataset )"""

    def __init__(self, num_points=1000, bias=2, alpha=3 / 4, sigma=0.1):
        super(_MyDataset, self).__init__()
        self.num_points = num_points
        self.b = torch.tensor(bias)
        self.alpha = torch.tensor(alpha)
        self.sigma = torch.tensor(sigma)
        x1 = -self.b + self.sigma * torch.randn((int(self.alpha * self.num_points),))
        x2 = self.b + self.sigma * torch.randn(
            (int((1 - self.alpha) * self.num_points),)
        )
        self.data = torch.concat([x1, x2])

    def __len__(self):
        return self.num_points

    def __getitem__(self, index):
        return self.data[index]

    def get_full_data(self):
        return self.data

    def ufunc0(self, x):
        """true potential function"""

        return -torch.log(
            self.alpha * torch.exp(-((x + self.b) ** 2) / 2 / self.sigma**2)
            + (1 - self.alpha) * torch.exp(-((x - self.b) ** 2) / 2 / self.sigma**2)
        )

    def pdf0(self, x):
        """true probability density function"""

        return self.alpha * (
            1 / torch.sqrt(torch.tensor(2) * torch.pi) / self.sigma
        ) * torch.exp(-((x + self.b) ** 2) / 2 / self.sigma**2) + (1 - self.alpha) * (
            1 / torch.sqrt(torch.tensor(2) * torch.pi) / self.sigma
        ) * torch.exp(
            -((x - self.b) ** 2) / 2 / self.sigma**2
        )


class GaussFourDataset(Dataset):
    """Four modes of Gaussian Distribution, which is more complex."""

    def __init__(
        self,
        num_of_points=10000,
        means=[-3, -1, 1, 3],
        sigma=0.1,
        alpha=[1 / 8, 3 / 8, 3 / 8, 1 / 8],
    ):
        super(GaussFourDataset, self).__init__()
        self.num_of_points = num_of_points
        self.means = torch.tensor(means)
        self.sigma = torch.tensor(sigma)
        self.alpha = alpha
        self.data = self._data_init()

    def _data_init(self):
        assert len(self.alpha) == self.means.shape[0]  # 满足一一对应
        assert abs(sum(self.alpha) - 1) < 0.01  # 满足和为1

        result = torch.tensor([])
        for i in range(len(self.alpha)):
            x = self.means[i] + self.sigma * torch.randn(
                (
                    int(
                        self.alpha[i] * self.num_of_points,
                    )
                )
            )
            result = torch.concat([result, x])
        return result

    def __len__(self):
        return self.num_of_points

    def __getitem__(self, index):
        return self.data[index]

    def get_full_data(self):
        return self.data

    def pdf0(self, x):
        """True probability density function

        It is ugly, but useful.
        """
        means = self.means
        stds = [self.sigma] * len(self.means)
        weights = self.alpha

        gaussian_distributions = [
            norm(loc=mean, scale=std) for mean, std in zip(means, stds)
        ]
        distribution_values = torch.zeros_like(x)
        for i, gaussian_dist in enumerate(gaussian_distributions):
            distribution_values += weights[i] * gaussian_dist.pdf(x)
        return distribution_values

    def ufunc0(self, x):
        return -torch.log(self.pdf0(x))


class GenerateDistributionOfT:
    """Generate samples of t according to different methods"""

    def __init__(
        self,
        method: Literal["uniform", "exponential", "normal"] = "uniform",
        time_dim: int = 10,
        device="cpu",
    ):
        """

        Args:
            method (str): uniform, expotential, normal
            time_dim (int): dimesions of timesteps
        """
        self._method = method
        self._time_dim = time_dim
        self.device = device
        self._probs = self._generate_probs()

    def sample_t(self, num_of_samples: int):
        return torch.multinomial(
            self._probs, num_samples=num_of_samples, replacement=True
        ).unsqueeze(-1)

    def _generate_probs(self):

        match self._method:
            case "uniform":
                probs = torch.ones(self._time_dim, device=self.device)
            case "exponential":
                probs = torch.exp(
                    -torch.arange(self._time_dim, device=self.device) / self._time_dim
                )
            case "normal":
                mean, sigma = 0.0, self._time_dim
                probs = torch.tensor(
                    norm.pdf(torch.arange(self._time_dim), mean, sigma),
                    device=self.device,
                )
            case _:
                raise NotImplementedError
        return probs
