import torch

class ReplayBufferEBM:
    """its size should be the same as the training set

    there have two random methods to init:
        1. uniform
        2. normal

    """

    def __init__(self, buffer_size=1e3, random_type="uniform"):
        self.buffer_size = buffer_size
        self.random_type = random_type
        self.buffer_of_samples = self.init_buffer()

    def init_buffer(self):
        """initialize the buffer with different type"""

        if self.random_type == "uniform":
            data = torch.zeros((self.buffer_size, 1)).uniform_(0, 1)
        elif self.random_type == "normal":
            data = torch.zeros((self.buffer_size, 1)).normal_(0, 1)
        else:
            raise NotImplementedError
        return data

    def sample_from_buffer(self, n_samples):
        """sample from buffer"""

        idx = torch.randint(low=0, high=self.buffer_size, size=(n_samples,))
        samples = self.buffer_of_samples[idx]

        return samples, idx

    def update_buffer(self, idx, samples):
        self.buffer_of_samples[idx] = samples

class ReplayBufferDAEBM:
    """Build a buffer to store (x,t),
    whose size is the same as that of training set.

    NOTE: The meaning of "labels" is the same as "t"

    """

    def __init__(self, buffer_size, element_shape, n_timesteps, random_type="normal"):
        """init the buffer

        Args:
            buffer_size:    size of the buffer
            element_shape:  tuple of shape, if one dimension, element_shape=(1,)
                            elif image, element_shape=(channels, height, width)
            n_timesteps:    maximum diffusion steps
            random_type:    init type, can be uniform or normal (default normal)

        """
        self.buffer_size = buffer_size
        self.element_shape = element_shape
        self.randam_type = random_type
        self.n_of_timesteps = n_timesteps

        # below: init data and labels in the buffer
        self.buffer_of_samples = self._init_buffer_data(self.buffer_size)
        self.buffer_of_t = self._init_buffer_labels()

    def _init_buffer_data(self, n_samples):
        """Init data in buffer (only x)"""
        if self.randam_type == "uniform":
            data = torch.zeros((n_samples,) + self.element_shape).uniform_(0, 1)
        elif self.randam_type == "normal":
            data = torch.zeros((n_samples,) + self.element_shape).normal_(0, 1)
        else:
            raise NotImplementedError

        return data

    def _init_buffer_labels(self):
        """Init labels in buffer (only t)

        Results is [0,0,0,...,1,1,1,...,T-1,T-1,T-1]
        For example, if buffer_size=10, n_of_timesteps=5,
        results is [0,0,0,1,1,1,2,2,2,3]

        NOTE: I think it is not so reasonable, or maybe initialization
        of t is not important. (Maybe change later)

        """

        return torch.arange(self.n_of_timesteps).repeat_interleave(
            int(self.buffer_size // self.n_of_timesteps) + 1
        )[: self.buffer_size]

    def sample_buffer(self, n_samples, reinit_probs=0.0, deterministic=False):
        """Sample from the buffer.

        Args:
            n_samples:      int, num of required samples.
            reinit_probs:   float, [0~1], whether to reset some samples to
                            noise, with probability of reinit_probs.
            deterministic:  bool, if true, return idx of [0:n_samples-1],
                                else, random choose. (default false)


        """
        idx = (
            torch.randint(0, self.buffer_size, (n_samples,))
            if not deterministic
            else torch.arange(n_samples)
        )
        samples = self.buffer_of_samples[idx]
        t = self.buffer_of_t[idx].unsqueeze(-1)

        if reinit_probs > 0:
            choose_random = torch.rand(n_samples) < reinit_probs
            samples[choose_random,] = self._init_buffer_data(sum(choose_random))
            t[choose_random] = (
                torch.zeros(sum(choose_random)).fill_(self.n_of_timesteps - 1).long()
            )

        return samples, t, idx

    def update_buffer(self, idx, samples, t):
        """Update buffer."""
        self.buffer_of_samples[idx] = samples
        self.buffer_of_t[idx] = t.squeeze()
