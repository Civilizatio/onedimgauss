import torch
import numpy as np

class GaussianDiffusion:
    """Get diffusion datas of x

    using x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) + epsilon_t

    """

    def __init__(
        self, n_timesteps=1000, beta_schedule="linear", beta_start=1e-5, beta_end=1e-2, device="cpu"
    ):
        self.n_timesteps = n_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = self._make_beta_schedule()
        self.betas = torch.cat([torch.zeros(1), self.betas.clamp_(0, 1)], 0)
        # above: add beta_0 = 0 for convenience

        self.device = torch.device(device)

        (
            self.sigmas,
            self.alphas,
            self.alphas_bar_sqrt,
            self.alphas_bar_comp_sqrt,
        ) = self._make_sigma_schedule()


        # Put tensors onto self.device
        (
            self.betas,
            self.sigmas,
            self.alphas,
            self.alphas_bar_sqrt,
            self.alphas_bar_comp_sqrt
        ) = (
            self.betas.to(self.device),
            self.sigmas.to(self.device),
            self.alphas.to(self.device),
            self.alphas_bar_sqrt.to(self.device),
            self.alphas_bar_comp_sqrt.to(self.device)
        )

    def q_sample(self, x_0, t, noise=None):
        """Get samples of x_t.

        Using:
            x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) + epsilon_t
        x_0 should be [Batch_size, Channels, Height, Width]
        t should be [Batch_size, 1]
        epsilon should be noise of N(0,I), size like x_0

        Returns:
            x_t: should be the same as x_0


        """

        if noise is None:
            # Default noise, device the same as x_0
            noise = torch.randn_like(x_0)
        xshape = x_0.shape
        alphas_bar_sqrt = self._extract(self.alphas_bar_sqrt, t, xshape)
        alphas_bar_comp_sqrt = self._extract(self.alphas_bar_comp_sqrt, t, xshape)
        x_t = alphas_bar_sqrt * x_0 + alphas_bar_comp_sqrt * noise
        return x_t

    def q_sample_progressive(self, x_0):
        """Generate a full sequence of disturbed data
        
        Return x_seq for plot """
        x_seq = []
        x_t = x_0
        for t in range(self.n_timesteps + 1):
            t_now = torch.ones(x_0.shape[0]).fill_(t).long().unsqueeze(-1).to(x_0.device)
            x_t = self._extract(self.alphas, t_now, x_t.shape) * x_t + self._extract(
                self.sigmas, t_now, x_t.shape
            ) * torch.randn_like(x_0)
            x_seq.append(x_t.squeeze())
        x_seq = torch.stack(x_seq, dim=0)
        return x_seq.cpu()  # NOTE: shoule be [self.n_timesteps+1, x_0.shape]

    def get_sigmas(self):
        return self.sigmas

    def _extract(self, input, t, xshape):
        """Extract some coefficients at specfied timesteps,
        then reshape to [batch_size, 1, 1, ...] for broadcasting purposes

        Size of imput should be [num_of_timesteps+1]
        Size of t should be [batch_size, 1]
        neet to gather input_t to [batch_size,1,1,...], the same like xshape,
        which should be [batch_size, channels, height, width]
        """
        out = torch.gather(input, 0, t.squeeze().to(input.device))
        reshape = [xshape[0]] + [1] * (len(xshape) - 1)
        return out.reshape(*reshape)

    def _make_beta_schedule(self):
        """get beta_t, t from 1 to T

        Args:
            :schedule: type of beta schedule, linear by default
            :n_timesteps: T
            :start: beta_1
            :end: beta_T
        Returns:
            :betas: size should be torch.Size([T])
        Raises:
            NotImplementedError

        """
        match self.beta_schedule:
            case "linear":
                betas = torch.linspace(
                    start=self.beta_start, end=self.beta_end, steps=self.n_timesteps
                )
            case "sigmoid":
                betas = torch.linspace(-6, 6, self.n_timesteps)
                betas = (
                    torch.sigmoid(betas) * (self.beta_end - self.beta_start)
                    + self.beta_start
                )
            case "sqrtlinear":
                betas = (
                    torch.linspace(self.beta_start, self.beta_end, self.n_timesteps)
                    ** 2
                )
            case "sqrtcumlinear":
                betas = (
                    torch.cumsum(
                        torch.linspace(
                            self.beta_start, self.beta_end, self.n_timesteps
                        ),
                        0,
                    )
                    ** 2
                )
            case "sqrtlog":
                betas = (
                    torch.logspace(-self.beta_start, -self.beta_end, self.n_timesteps)
                    ** 2
                )
            case "quad":
                betas = (
                    torch.linspace(
                        self.beta_start**0.5, self.beta_end**0.5, self.n_timesteps
                    )
                    ** 2
                )
            case "geometric":
                betas = torch.tensor(
                    np.exp(
                        np.linspace(
                            np.log(self.beta_start),
                            np.log(self.beta_end),
                            self.n_timesteps,
                        )
                    )
                ).float()
            case _:
                raise NotImplementedError
        return betas

    def _make_sigma_schedule(self):
        """ Get the noise level schedule"""
        sigmas = torch.sqrt(self.betas)  # for add Gaussian noise
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        alphas_bar_sqrt = torch.sqrt(alphas_bar)
        alphas_bar_comp_sqrt = torch.sqrt(1 - alphas_bar)

        return sigmas, alphas, alphas_bar_sqrt, alphas_bar_comp_sqrt