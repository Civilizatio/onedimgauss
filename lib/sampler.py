import torch
import torch.nn.functional as F

class MALA_Sampling:
    """ Metropolis-Adjusted Langevin Algorithm. """

    def __init__(self, is_reject=True, mala_sigma=0.1, mala_n=10, device="cpu"):
        self.is_reject = is_reject
        self.sigma = mala_sigma # only use one sigma
        self.n = mala_n  # num of steps
        self.device = device

        self.acpt_rate = None

    def __call__(self, net, init_samples):
        return self.generate_samples(net, init_samples)
    
    def mala_sampling(self, net, init_samples):
        return self.generate_samples(net, init_samples)

    def _cal_energies_and_grads(self, net, samples):
        """Get energies and grad of energies of samples"""

        U = net(samples)
        U_grad = torch.autograd.grad(U.sum(), [samples], retain_graph=False)[0]
        return U, U_grad

    def _cal_tran_probs(
        self,
        U,
        U_prop,
        U_grad,
        U_prop_grad,
        noise,  # [B,1]
    ):
        """if reject: prob = min(1, r=p(y)p(y|x)/p(x)p(x|y))
        else prob = 1

        r = exp(U_0-U_1+1/2*noise^2-1/2*(-noise+sigma/2*(U_0_grad+U_1_grad)))


        """

        if self.is_reject:
            with torch.no_grad():
                trans_probs = torch.exp(
                    U
                    - U_prop
                    + 0.5 * torch.pow(noise, 2)
                    - 0.5
                    * torch.pow(-noise + self.sigma / 2 * (U_grad + U_prop_grad), 2)
                )

                trans_probs = torch.min(
                    trans_probs, torch.ones_like(trans_probs).to(self.device)
                )

        else:
            trans_probs = torch.ones(U.shape[0]).unsqueeze(-1).to(self.device)

        return trans_probs

    def generate_samples(
        self,
        net,
        init_samples,
    ):
        """ generate samples using MALA, return samples after {self.n} steps"""

        self.acpt_rate = torch.zeros_like(init_samples)
        samples = torch.autograd.Variable(init_samples.clone(), requires_grad=True).to(
            self.device
        )
        prop_samples = torch.autograd.Variable(
            init_samples.clone(), requires_grad=True
        ).to(self.device)

        U, U_grad = self._cal_energies_and_grads(net, samples)

        for _ in range(self.n):
            noise = torch.randn_like(samples)
            prop_samples.data = (
                samples.data - self.sigma / 2 * U_grad + self.sigma * noise
            )

            # whether to accept or reject
            U_prop, U_prop_grad = self._cal_energies_and_grads(net, prop_samples)
            trans_probs = self._cal_tran_probs(U, U_prop, U_grad, U_prop_grad, noise)

            acpt_or_not = torch.rand_like(trans_probs).to(self.device) < trans_probs
            self.acpt_rate += acpt_or_not

            # update samples
            samples.data[acpt_or_not] = prop_samples.data[acpt_or_not]

            # update energies and grads
            U.data[acpt_or_not] = U_prop.data[acpt_or_not]
            U_grad.data[acpt_or_not] = U_prop_grad.data[acpt_or_not]

        # final samples
        final_samples = samples.detach().clone().cpu()

        # accept rate
        self.acpt_rate = self.acpt_rate / self.n

        return final_samples

class MGMS_sampling:
    """Langevin dynamic sampling with reject (MALA within Gibbs mixture sampling)

    Given current observation (x_0, t_0), and num_steps,
    along with step size sigma_t, which is the same as diffusion
    sigma_t = sqrt{beta_t}
    """

    def __init__(self, num_steps, init_step_size, is_reject=True, device="cpu"):
        self.num_steps = num_steps  # steps of Langevin transpose
        self.is_reject = is_reject
        self.acpt_rate = 0
        self.device = torch.device(device)
        self.init_step_size = init_step_size.to(self.device)

    def mgms_sampling(self, net, x, t):
        """Given (x,t), return samples

        Args:
            net:    calculate energy function
            x:      size should be [batch_size, channels, height, weight]
            t:      size should be [batch_size, 1]
        """

        # acpt_or_reject_list: record accept or not of different x of each step
        acpt_or_reject_list = torch.zeros(self.num_steps, x.shape[0])
        step_size = self._extract(self.init_step_size, t, x.shape)
        step_size_square = torch.pow(step_size, 2)

        x_t = torch.autograd.Variable(x.clone(), requires_grad=True)
        x_t_prop = torch.autograd.Variable(x.clone(), requires_grad=True)

        U = net.energy_output(x_t, t)
        U_grad = torch.autograd.grad(U.sum(), [x_t], retain_graph=False)[0]

        for i in range(self.num_steps):
            noise = torch.randn_like(x_t)

            # below: Langevin update
            # x_{t+1} = x_t - sigma^2/2 \nabla U(x_t) + sigma * epsilon
            x_t_prop.data = (
                x_t.data - 0.5 * step_size_square * U_grad + step_size * noise
            )

            U_prop = net.energy_output(x_t_prop, t)
            U_prop_grad = torch.autograd.grad(
                U_prop.sum(), [x_t_prop], retain_graph=False
            )[0]

            # below: reject or not
            trans_probs = self._cal_tran_probs(
                U, U_prop, U_grad, U_prop_grad, step_size_square, noise
            )
            acpt_or_not = torch.rand_like(trans_probs) < trans_probs

            acpt_or_reject_list[i] = acpt_or_not.squeeze().cpu()

            # below update x, U, U_grad
            x_t.data[acpt_or_not,] = x_t_prop.data[acpt_or_not,]
            U.data[acpt_or_not] = U_prop.data[acpt_or_not]
            U_grad.data[acpt_or_not,] = U_prop_grad.data[acpt_or_not,]

        # Calculate accept rate of different t
        # This is a little strange.

        self.acpt_rate = torch.tensor(
            [
                acpt_or_reject_list[:, t.squeeze().cpu() == j].float().nanmean()
                for j in range(len(self.init_step_size))
            ]
        )

        # Below: sample t_1 with p(t_1|x_1)
        with torch.no_grad():
            output = torch.zeros(x_t.shape[0], net.n_class, device=self.device)
            for i in torch.arange(net.n_class):
                t_i = torch.full(
                    (x_t.shape[0], 1), i, device=x_t.device, dtype=torch.long
                )
                output[:, i] = -net.energy_output(x_t, t_i).squeeze()
            t_probs = F.softmax(output, dim=1)
        # print(f"Time of global jump is {timer_softmax.elapsed_time}s")

        # get t: [batch_size, 1]
        t = torch.multinomial(t_probs, num_samples=1, replacement=True)

        return x_t.detach().cpu(), t.cpu(), self.acpt_rate

    def adjust_step_size_given_acpt_rate(self, delta=0.2):

        init_step_size = self.init_step_size.cpu()
        self.init_step_size = torch.where(
            self.acpt_rate > 0.8,
            torch.tensor(1 + 0.5 * delta) * init_step_size,
            torch.where(
                self.acpt_rate < 0.6,
                init_step_size / (1 + delta),
                init_step_size,
            ),
        ).to(self.device)

    def get_init_step_size(self):
        return self.init_step_size

    def get_isreject(self):
        return self.is_reject

    def update_isreject(self, reject: bool):
        self.is_reject = reject

    def _cal_tran_probs(
        self,
        U,
        U_prop,
        U_grad,
        U_prop_grad,
        step_size_square,
        noise,  # [B,1]
    ):
        """if reject: prob = min(1, r=p(y)p(y|x)/p(x)p(x|y))
        else prob = 1

        r = exp(U_0-U_1+1/2*noise^2-1/2*(-noise+sigma/2*(U_0_grad+U_1_grad)))


        """

        if self.is_reject:
            with torch.no_grad():
                trans_probs = torch.exp(
                    U
                    - U_prop
                    + 0.5 * torch.pow(noise, 2)
                    - 0.5
                    * torch.pow(
                        -noise + step_size_square / 2 * (U_grad + U_prop_grad), 2
                    )
                )

                trans_probs = torch.min(trans_probs, torch.ones_like(trans_probs))

        else:
            trans_probs = torch.ones(U.shape[0], device=self.device).unsqueeze(-1)

        return trans_probs

    def _extract(self, input, t, xshape):
        """ Extract some coefficients at specfied timesteps,
        then reshape to [batch_size, 1, 1, ...] for broadcasting purposes

        Size of imput should be [num_of_timesteps+1]
        Size of t should be [batch_size, 1]
        neet to gather input_t to [batch_size,1,1,...], the same like xshape,
        which should be [batch_size, channels, height, width]
        """
        out = torch.gather(input, 0, t.squeeze().to(input.device))
        reshape = [xshape[0]] + [1] * (len(xshape) - 1)
        return out.reshape(*reshape)

class MGMS_sampling_LocalJump(MGMS_sampling):
    """ MGMS sampling with local jump."""

    def __init__(
        self, num_steps, init_step_size, is_reject=True, device="cpu", window_size=5
    ):
        super().__init__(num_steps, init_step_size, is_reject, device)
        self.window_size = window_size

    def mgms_sampling(self, net, x, t, local_jump_enabled: bool = False):
        """Given (x,t), return samples

        Using local jump instead of global jump.

        Args:
            net:
            x:
            t:
            local_jump_enabled(bool): whether to use global jump
        """

        if not local_jump_enabled:
            return super().mgms_sampling(net, x, t)

        # acpt_or_reject_list: record accept or not of different x of each step
        acpt_or_reject_list = torch.zeros(self.num_steps, x.shape[0])
        step_size = self._extract(self.init_step_size, t, x.shape)
        step_size_square = torch.pow(step_size, 2)

        x_t = torch.autograd.Variable(x.clone(), requires_grad=True)
        x_t_prop = torch.autograd.Variable(x.clone(), requires_grad=True)

        U = net.energy_output(x_t, t)
        U_grad = torch.autograd.grad(U.sum(), [x_t], retain_graph=False)[0]

        for i in range(self.num_steps):

            # Update x -> x_1
            noise = torch.randn_like(x_t)

            # below: Langevin update
            # x_{t+1} = x_t - sigma^2/2 \nabla U(x_t) + sigma * epsilon
            x_t_prop.data = (
                x_t.data - 0.5 * step_size_square * U_grad + step_size * noise
            )

            U_prop = net.energy_output(x_t_prop, t)
            U_prop_grad = torch.autograd.grad(
                U_prop.sum(), [x_t_prop], retain_graph=False
            )[0]

            # below: reject or not
            trans_probs = self._cal_tran_probs(
                U, U_prop, U_grad, U_prop_grad, step_size_square, noise
            )
            acpt_or_not = torch.rand_like(trans_probs) < trans_probs

            acpt_or_reject_list[i] = acpt_or_not.squeeze().cpu()

            # below update x, U, U_grad
            x_t.data[acpt_or_not,] = x_t_prop.data[acpt_or_not,]
            U.data[acpt_or_not] = U_prop.data[acpt_or_not]
            U_grad.data[acpt_or_not,] = U_prop_grad.data[acpt_or_not,]

        self.acpt_rate = torch.tensor(
            [
                acpt_or_reject_list[:, t.squeeze().cpu() == j].float().nanmean()
                for j in range(len(self.init_step_size))
            ]
        )

        # Update t -> t_1
        # step 1: prop t_1/2

        new_idx = torch.randint_like(t, high=self.window_size)
        t_prop = (
            new_idx
            + torch.clamp(
                t,
                min=self.window_size // 2,
                max=net.n_class - self.window_size // 2 - 1,
            )
            - self.window_size // 2
        )

        # Calculate transpose probability
        with torch.no_grad():
            U_prop = net.energy_output(x_t, t_prop)
        trans_probs = self._cal_tran_probs_of_t(U, U_prop)
        acpt_or_not = torch.rand_like(trans_probs) < trans_probs
        t.data[acpt_or_not] = t_prop.data[acpt_or_not]
        # print(f"Time of local jump is {timer_t_proposal.elapsed_time}s")
        return x_t.detach().cpu(), t.cpu(), self.acpt_rate

    def _cal_tran_probs_of_t(self, U, U_prop):
        """Calculate transpose probability

        prob = min{1, r}
        r = exp{U(x_1,t_0)-U(x_1,t_1/2)}

        NOTE:
            If put local jump into process of langevin dynamic,
            you need to update U and U_prop in the mgms_sampling function.
        """
        if self.is_reject:
            with torch.no_grad():
                trans_probs = torch.exp(U - U_prop)
                trans_probs = torch.min(trans_probs, torch.ones_like(trans_probs))
        else:
            trans_probs = torch.ones((U.shape[0], 1), device=self.device)
        return trans_probs
