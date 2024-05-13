import torch
import matplotlib.pyplot as plt
import os


def plot_energy_and_density_of_diffusion_times(
    dataset, net, device, num_diffusion_timesteps
):
    
    """Plot energy functions and density functions of different timesteps.

    NOTE Density functions there of timesteps not equal to zero are not
    calculated, for it is a little difficult. And this is the same method as DAEBM
    """
    x = torch.linspace(-4, 4, 100).unsqueeze(-1)
    energy_truth = dataset.ufunc0(x).squeeze()
    density_truth = dataset.pdf0(x).squeeze()
    energy_truth_normalized = energy_truth - energy_truth.min()
    density_truth_normalized = torch.exp(-energy_truth_normalized)

    fig, ax = plt.subplots(
        num_diffusion_timesteps + 1,
        4,
        figsize=(10, 2.5 * (num_diffusion_timesteps + 1)),
    )
    for idx in range(num_diffusion_timesteps + 1):
        energy_model = (
            net.energy_output(
                x.to(device), torch.zeros((x.shape[0], 1)).fill_(idx).long().to(device)
            )
            .detach()
            .clone()
            .squeeze()
            .cpu()
        )
        density_model = torch.exp(-energy_model)
        energy_model_normalized = energy_model - energy_model.min()
        density_model_normalized = torch.exp(-energy_model_normalized)

        ax[idx][0].plot(
            x.squeeze(),
            density_truth,
            linewidth=2,
            markersize=12,
            label="True",
            color="#ff7f0e",
        )

        ax[idx][0].plot(
            x.squeeze(), density_model, linewidth=2, markersize=12, label="Learned"
        )

        ax[idx][0].set_title(f"Unnormalized Density Time {idx}")

        ax[idx][1].plot(
            x.squeeze(),
            density_truth_normalized,
            linewidth=2,
            markersize=12,
            label="True",
            color="#ff7f0e",
        )

        ax[idx][1].plot(
            x.squeeze(),
            density_model_normalized,
            linewidth=2,
            markersize=12,
            label="Learned",
        )

        ax[idx][1].set_title(f"Normalized Density Time {idx}")

        ax[idx][2].plot(
            x.squeeze(),
            energy_truth,
            linewidth=2,
            markersize=12,
            label="True",
            color="#ff7f0e",
        )

        ax[idx][2].plot(
            x.squeeze(), energy_model, linewidth=2, markersize=12, label="Learned"
        )

        ax[idx][2].set_title(f"Unnormalized Energy Time {idx}")

        ax[idx][3].plot(
            x.squeeze(),
            energy_truth_normalized,
            linewidth=2,
            markersize=12,
            label="True",
            color="#ff7f0e",
        )

        ax[idx][3].plot(
            x.squeeze(),
            energy_model_normalized,
            linewidth=2,
            markersize=12,
            label="Learned",
        )

        ax[idx][3].set_title(f"Normalized Energy Time {idx}")
        handles, labels = ax[idx][3].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.15, 0.1))

    plt.tight_layout()
    plt.close()

    return fig


def plot_results_of_ebm(dataset, net, replay_buffer):
    """Plot results of EBM training.

    Include energy function, replay_buffer distribution,
    long_run results, post-sampling results.
    """

    ## results of post training
    fig, axes = plt.subplots(2, 2, sharex=True)
    x = torch.linspace(-4, 4, 100).unsqueeze(-1)

    # energy function
    energy_truth = dataset.ufunc0(x).squeeze()
    energy_model = net(x).detach().squeeze()
    axes[0, 0].plot(
        x.squeeze(), energy_model - torch.min(energy_model), color="red", label="model"
    )
    axes[0, 0].plot(
        x.squeeze(),
        energy_truth - torch.min(energy_truth),
        color="blue",
        label="ground truth",
    )
    axes[0, 0].set_title("Energy function")
    axes[0, 0].set_ylabel("Energy")
    axes[0, 0].legend()

    # Replay buffer
    y, _ = replay_buffer.sample_from_buffer(dataset.num_of_points)
    y = y.squeeze()
    y_truth = dataset.pdf0(x.clone().detach()).squeeze()
    axes[0, 1].hist(y, bins=100, range=(-4, 4), density=True)
    axes[0, 1].plot(x, y_truth, color="red", label="true density")
    axes[0, 1].set_title("Replay buffer")
    axes[0, 1].legend()

    # Long run samples
    from oneD_Gaussian import MALA_Sampling

    mala_sampler_tmp = MALA_Sampling(mala_n=1000)
    long_run_samples = mala_sampler_tmp(
        net, dataset.get_full_data().unsqueeze(-1)
    ).squeeze()
    axes[1, 0].hist(long_run_samples, bins=100, range=(-4, 4), density=True)
    axes[1, 0].plot(x, y_truth, color="red", label="true density")
    axes[1, 0].set_title("Long-run Samples")
    axes[1, 0].legend()

    # Post sampling
    post_init_samples = torch.randn((dataset.num_of_points, 1))
    post_sampling_samples = mala_sampler_tmp(net, post_init_samples).squeeze()
    axes[1, 1].hist(post_sampling_samples, bins=100, range=(-4, 4), density=True)
    axes[1, 1].plot(x, y_truth, color="red", label="true density")
    axes[1, 1].set_title("Post-training Samples")
    axes[1, 1].legend()

    return fig


def plot_energy_function(
    dataset, net, is_save=False, file_prefix="./figures/", filename="energy_results.png"
):
    x = torch.linspace(-4, 4, 100).unsqueeze(-1)

    energy0 = dataset.ufunc0(x).squeeze()
    energy_model = net(x).detach().squeeze()

    fg = plt.figure(figsize=(8, 5))
    plt.plot(
        x.squeeze(), energy_model - torch.min(energy_model), color="red", label="model"
    )
    plt.plot(
        x.squeeze(), energy0 - torch.min(energy0), color="blue", label="ground truth"
    )
    plt.xlim((-4, 4))
    plt.xlabel("X")
    plt.ylabel("Energy")
    plt.legend()

    if is_save:
        if not os.path.exists(os.path.dirname(file_prefix)):
            os.makedirs(os.path.dirname(file_prefix))
        fg.savefig(file_prefix + filename)

    return fg

