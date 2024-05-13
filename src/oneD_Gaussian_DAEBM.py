# DAEBM中一维高斯实验的DAEBM实现
# Date:2024/02/20 changed by Li Ke,
# Comment:  1. 增加 TensorBoard 模块，统计过程中出现的图像、数据；
#           2. loss统计正相以及负相，增加到 TensorBoard 中。
#           3. 每个数个 epoch 统计不同状态下采样状态的分布
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
import logging
from logging import Logger
import matplotlib.pyplot as plt
from typing import Union, Dict
import seaborn as sn
import os
import shutil
import numpy as np
from torch.utils.tensorboard import SummaryWriter

try:
    from lib.config_parser import ParserUtils
    from lib.create_dataset import GaussFourDataset, GenerateDistributionOfT
    from lib.diffusion import GaussianDiffusion
    from lib.model import MyUfuncTemb
    from lib.sampler import MGMS_sampling, MGMS_sampling_LocalJump
    from lib.replay_buffer import ReplayBufferDAEBM
    from lib.train_utils import (
        InitializeNet,
        OptimizerConfigure,
        SchedulerConfigure,
        save_checkpoint,
    )
    from lib.utils import Timer, Accumulator, AverageMeter
    from lib.plot_utils import (
        plot_results_of_ebm,
        plot_energy_and_density_of_diffusion_times,
    )
except ImportError:
    raise


####### Some tool functions
def training_losses(net, x_pos, t_pos, x_neg, t_neg):
    """Calculate training losses using ML.

    loss = U_pos - U_neg
    """
    energy_pos = net.energy_output(x_pos, t_pos)
    energy_neg = net.energy_output(x_neg, t_neg)

    loss_pos = energy_pos.mean()
    loss_neg = energy_neg.mean()

    loss = loss_pos - loss_neg

    return loss, loss_pos, loss_neg


def train(
    net: MyUfuncTemb,
    replay_buffer: ReplayBufferDAEBM,
    dataset: Dataset,
    dataloader: DataLoader,
    optimizer: Union[Adam, SGD],
    batch_size: int,
    scheduler: Union[MultiStepLR, LambdaLR],
    warmup_scheduler: SchedulerConfigure.WarmUpLR,
    mala_sampler: MGMS_sampling_LocalJump,
    t_sampler: GenerateDistributionOfT,
    gauss_diffusion: GaussianDiffusion,
    logger: Logger,
    writer: SummaryWriter,
    device: torch.device,
    accumulators: Dict[str, Accumulator],
    averagemeters: Dict[str, AverageMeter],
    dynamic_sampling: bool,
    local_jump_enabled: bool,
    start_local_epoch: int,
    num_diffusion_timesteps: int,
    num_of_points: int = 1000,
    num_of_epochs: int = 200,
    n_warm_epochs: int = 10,
    start_reject_epochs: int = 10,
    print_freq: int = 9,
):
    losses = []
    niter = 0
    time0_bank = torch.randn((num_of_points, 1))  # Record data of t=0
    time0_save_sample_idx = 0
    iter_per_epoch = num_of_points // batch_size

    # Record time
    timer_list = ["epoch", "sampling"]
    timers = {key: Timer() for key in timer_list}

    for epoch in range(num_of_epochs):

        timers["epoch"].start()
        mgms_local_enabled = local_jump_enabled and epoch >= start_local_epoch
        for idx in range(iter_per_epoch):
            niter += 1

            # Init x_t and t
            t = t_sampler.sample_t(num_of_samples=x_0.shape[0])
            x_0 = next(iter(dataloader)).unsqueeze(-1)
            x_t = gauss_diffusion.q_sample(x_0.to(device), t)

            init_x_t_neg, init_t_neg, buffer_idx = replay_buffer.sample_buffer(
                n_samples=batch_size
            )

            # Get x_neg, t_neg through MGMS sampling
            x_t_neg, t_neg, acpt_rate = mala_sampler.mgms_sampling(
                net, init_x_t_neg.to(device), init_t_neg.to(device), mgms_local_enabled
            )

            # Update accept rate of MGMS
            accumulators["mala_acpt_rate"].add(acpt_rate.nan_to_num())

            # Count the timesteps
            labels_accumulator = [
                torch.sum(t_neg == i).item() for i in range(num_diffusion_timesteps + 1)
            ]
            accumulators["labels"].add(labels_accumulator)

            # Count for label transfer
            jump_mat = np.zeros(
                ((args.num_diffusion_timesteps + 1), (args.num_diffusion_timesteps + 1))
            )
            
            jump_coordinates = (
                torch.cat([init_t_neg.view(-1, 1), t_neg.view(-1, 1).cpu()], 1).cpu().numpy()
            ) # [B] [B] -> [B,2]
            np.add.at(jump_mat, tuple(zip(*jump_coordinates)), 1)
            accumulators["labels_jump_mat"].add(jump_mat.reshape(-1))

            t_neg = t_neg.to(device)
            x_t_neg = x_t_neg.to(device)

            # Calculate loss
            loss, loss_pos, loss_neg = training_losses(
                net=net, x_pos=x_t, t_pos=t, x_neg=x_t_neg, t_neg=t_neg
            )

            if torch.isnan(loss) or loss.abs().item() > 1e8:
                logger.error("Training breakdown")
                break

            # Update parameters
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # Acquire norm of grads of loss
            grads_norms = torch.tensor(
                [param.grad.norm().item() for param in net.parameters()]
            )
            grad_norm_sum = torch.norm(grads_norms)

            # Acquire norm of grads of loss_pos and loss_neg
            gradients = torch.autograd.grad(
                loss_pos, net.parameters(), retain_graph=True
            )
            pos_grad_norm = torch.norm(
                torch.tensor([grad.norm().item() for grad in gradients])
            )

            gradients = torch.autograd.grad(
                loss_neg, net.parameters(), retain_graph=True
            )
            neg_grad_norm = torch.norm(
                torch.tensor([grad.norm().item() for grad in gradients])
            )

            optimizer.step()
            losses.append(loss.item())

            averagemeters["loss"].update(loss.item(), batch_size)
            averagemeters["loss_pos"].update(loss_pos.item(), batch_size)
            averagemeters["loss_neg"].update(loss_neg.item(), batch_size)
            averagemeters["loss_grad_norm"].update(grad_norm_sum.item(), batch_size)
            averagemeters["pos_grad_norm"].update(pos_grad_norm.item(), batch_size)
            averagemeters["neg_grad_norm"].update(neg_grad_norm.item(), batch_size)

            writer.add_scalars(
                main_tag="Loss_iter",
                tag_scalar_dict={
                    "Loss": loss.item(),
                    "Loss_pos": loss_pos.item(),
                    "Loss_neg": loss_neg.item(),
                },
                global_step=niter,
            )
            writer.add_scalars(
                main_tag="Loss_grad_iter",
                tag_scalar_dict={
                    "loss_grad_norm": grad_norm_sum,
                    "pos_grad_norm": pos_grad_norm,
                    "neg_grad_norm": neg_grad_norm,
                },
                global_step=niter,
            )

            # Update replay buffer
            t_neg = t_neg.cpu()
            x_t_neg = x_t_neg.cpu()
            replay_buffer.update_buffer(buffer_idx, x_t_neg, t_neg)

            # Time0 samples
            data_samples = x_t_neg  # [B,1]
            data_labels = t_neg  # [B,1]
            time0_samples_idx = data_labels == 0
            time0_samples = data_samples[time0_samples_idx]

            fid_slice = slice(
                time0_save_sample_idx % time0_bank.shape[0],
                min(
                    time0_bank.shape[0], time0_save_sample_idx + time0_samples.shape[0]
                ),
            )
            time0_save_sample_idx = (
                time0_save_sample_idx + time0_samples.shape[0]
            ) % time0_bank.shape[0]
            time0_bank[fid_slice] = time0_samples[
                : (fid_slice.stop - fid_slice.start)
            ].unsqueeze(-1)

            if epoch < n_warm_epochs:
                warmup_scheduler.step()
            # Whether to print
            if idx % print_freq == 0:
                logger.info(
                    "Epoch: [{0}][{1}/{2}] "
                    "Loss {loss.val:.4f} ({loss.avg:.4f}) "
                    "acceptance rate: {acpt_rate:.4f} "
                    "lr {lr:.6} ".format(
                        epoch,
                        idx,
                        len(dataloader) - 1,
                        loss=averagemeters["loss"],
                        acpt_rate=torch.mean(
                            torch.tensor(accumulators["mala_acpt_rate"].average())
                        ),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
        else:
            # Learning rate update
            scheduler.step()

            # Record time of one epoch
            timers["epoch"].stop()
            logger.info(
                f"EPOCH:{epoch} - Using time of EPOCH is {timers['epoch'].elapsed_time}s"
            )

            # Average loss and grad
            writer.add_scalars(
                main_tag="Loss/Average",
                tag_scalar_dict={
                    "average_loss": averagemeters["loss"].avg,
                    "average_pos": averagemeters["loss_pos"].avg,
                    "average_neg": averagemeters["loss_neg"].avg,
                },
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Loss_grad/Average",
                tag_scalar_dict={
                    "average_loss_grad": averagemeters["loss_grad_norm"].avg,
                    "average_pos_grad": averagemeters["pos_grad_norm"].avg,
                    "average_neg_grad": averagemeters["neg_grad_norm"].avg,
                },
                global_step=epoch,
            )
            # Reset Averagemeters
            for key in averagemeters:
                averagemeters[key].reset()

            # Plot energy function
            if (epoch % (num_of_epochs // 10)) == 0 or (epoch == num_of_epochs - 1):
                x = torch.linspace(-4, 4, 100).unsqueeze(-1)
                energy_truth = dataset.ufunc0(x).squeeze()
                energy_model = (
                    net.energy_output(x.to(device)).detach().clone().squeeze().cpu()
                )
                fig = plt.figure(figsize=(8, 5))
                plt.plot(
                    x.squeeze(),
                    energy_model - torch.min(energy_model),
                    color="red",
                    label="model",
                )
                plt.plot(
                    x.squeeze(),
                    energy_truth - torch.min(energy_truth),
                    color="blue",
                    label="ground truth",
                )
                plt.ylabel("Energy")
                plt.legend()
                writer.add_figure("Energy function", fig, global_step=epoch)

                fig, axes = plt.subplots(
                    1, num_diffusion_timesteps + 1, figsize=(10, 8)
                )
                x = torch.linspace(-4, 4, 100).unsqueeze(-1)
                for i in range(num_diffusion_timesteps + 1):
                    t = torch.zeros((x.shape[0], 1)).fill_(i).long().to(device)
                    energy_model = (
                        net.energy_output(x.to(device), t)
                        .detach()
                        .clone()
                        .squeeze()
                        .cpu()
                    )
                    axes[i].plot(x.squeeze(), energy_model - torch.min(energy_model))
                    axes[i].set_title(f"Time {i}")
                writer.add_figure(
                    "Energy function of diffusion data", fig, global_step=epoch
                )

                # Count different timelabels in replay buffer
                fig, axes = plt.subplots(
                    1, num_diffusion_timesteps + 1, figsize=(10, 8)
                )
                for i in range(num_diffusion_timesteps + 1):
                    data = replay_buffer.buffer_of_samples[
                        replay_buffer.buffer_of_t == i
                    ]
                    if torch.numel(data) == 0:
                        data = torch.tensor([0])
                    axes[i].hist(data.squeeze(), bins=100, range=(-4, 4))
                plt.tight_layout()
                writer.add_figure("Replay buffer data", fig, global_step=epoch)

                # Plot energy function and density curve
                fig = plot_energy_and_density_of_diffusion_times(
                    dataset=dataset,
                    net=net,
                    device=device,
                    num_diffusion_timesteps=num_diffusion_timesteps,
                )
                writer.add_figure("Density Slice", fig, global_step=epoch)

            # Count the number of timesteps at every 25 epochs.
            # 
            if epoch % 25 == 0:
                data = torch.tensor(accumulators["labels"].data)
                data = data / (accumulators["labels"].len * batch_size)
                fig = plt.figure()
                plt.bar(torch.arange(num_diffusion_timesteps + 1), data)
                plt.xlabel("TimeSteps")
                plt.ylabel("Distribution")
                writer.add_figure(
                    "Distribution of timesteps over 25 epochs", fig, global_step=epoch
                )
                accumulators["labels"].reset()

            if epoch % 20 == 0:
                # Save models
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "step_size": mala_sampler.get_init_step_size(),
                    },
                    save_path_prefix="./results/",
                )

            # Whether to use reject when mgms sampling
            if (
                start_reject_epochs is not None
                and epoch == start_reject_epochs - 1
                and mala_sampler.get_isreject() is False
            ):
                logger.warning("Change Sampler to do proper sampling with rejection")
                mala_sampler.update_isreject(True)

            # Update step size of LD according to acpt_rate
            acpt_rate = accumulators["mala_acpt_rate"].average()
            if (
                dynamic_sampling is True
                and mala_sampler.get_isreject() is True
                and epoch >= start_reject_epochs
            ):
                mala_sampler.adjust_step_size_given_acpt_rate()

            # Record mala_acpt_rate
            accumulators["mala_acpt_rate"].reset()
            for i in range(num_diffusion_timesteps + 1):
                writer.add_scalar("AcptRate/Time" + str(i), acpt_rate[i], epoch)

    return losses, time0_bank


def main(args):

    # Dir config
    exp_dir = os.path.join(args.main_dir, args.exp_dir)
    try:
        os.makedirs(exp_dir)
    except FileExistsError:
        shutil.rmtree(exp_dir)
        os.makedirs(exp_dir)

    # Logging init
    logging.basicConfig(
        format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    logger = logging.getLogger("DAEBM Training")

    # Add fileHandler
    fh = logging.FileHandler(filename=f"{exp_dir}/log.txt", mode="w")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Summary Writer
    log_dir = os.path.join(exp_dir, args.log_dir)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # Random seed
    torch.manual_seed(args.t_seed)

    # Dataset
    peaks_ratio = [x / sum(args.peaks_ratio) for x in args.peaks_ratio]
    myDataset = GaussFourDataset(
        num_of_points=args.num_of_points, means=args.means, alpha=peaks_ratio
    )

    # DataLoader
    dataloader = DataLoader(
        myDataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # Networks Configuration
    net = MyUfuncTemb(
        in_ch=1,
        n_timesteps=args.num_diffusion_timesteps + 1,
        act_func=args.act_func,
        time_embedding_type=args.time_embedding_type,
    )
    logger.info(str(net))
    net = InitializeNet.initialize_net(net, args.net_init_method).to(device)

    # Optimizer Configuration
    optimizer = OptimizerConfigure.configure_optimizer(
        params=net.parameters(),
        optimizer_type=args.optimizer_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=args.betas,
        sgd_momentum=args.sgd_momentum,
    )

    # Scheduler Configuration
    scheduler = SchedulerConfigure.configure_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler_type,
        milestones=args.milestones,
        lr_decay_factor=args.lr_decay_factor,
        n_epochs=args.n_epochs,
    )
    warmup_scheduler = SchedulerConfigure.configure_warmup_scheduler(
        optimizer=optimizer,
        n_warm_iters=args.n_warm_epochs * len(dataloader),
    )

    # Sampling Configuration
    replay_buffer = ReplayBufferDAEBM(
        buffer_size=args.replay_buffer_size,
        element_shape=(1,),
        n_timesteps=args.num_diffusion_timesteps + 1,
        random_type=args.random_type,
    )

    # Diffusion Configuration
    gauss_diffusion = GaussianDiffusion(
        n_timesteps=args.num_diffusion_timesteps,
        beta_schedule=args.diffusion_schedule,
        beta_start=args.diffusion_betas[0],
        beta_end=args.diffusion_betas[1],
        device=device,
    )

    # Get init step size of Langevin Dynamic
    b_square = args.b_factor**2
    step_size_square = torch.pow(gauss_diffusion.get_sigmas(), 2) * b_square
    step_size_square[0] = step_size_square[1]
    init_step_size = step_size_square.sqrt()

    logger.info("sigmas:" + str(gauss_diffusion.get_sigmas()))
    logger.info("step_size:" + str(init_step_size))

    # Change MGMS_sampling to local jump version
    # @2024/03/25
    mala_sampler = MGMS_sampling_LocalJump(
        num_steps=args.sample_steps,
        init_step_size=init_step_size,
        is_reject=args.mala_isreject,
        device=device,
        window_size=args.window_size,
    )

    # t sampler
    t_sampler = GenerateDistributionOfT(
        method=args.sample_method_of_t,
        time_dim=args.num_diffusion_timesteps + 1,
        device=device,
    )

    # Ancillary classes
    accumulators = {}
    accumulators["mala_acpt_rate"] = Accumulator(args.num_diffusion_timesteps + 1)
    accumulators["labels"] = Accumulator(args.num_diffusion_timesteps + 1)
    accumulators["labels_jump_mat"] = Accumulator((args.num_diffusion_timesteps+1)**2)
    meter_list = [
        "loss",
        "loss_pos",
        "loss_neg",
        "loss_grad_norm",
        "pos_grad_norm",
        "neg_grad_norm",
    ]
    meters = {key: AverageMeter() for key in meter_list}

    # Draw points of different time steps.
    x_ = myDataset.get_full_data().unsqueeze(-1)
    diffuse_data_pool = gauss_diffusion.q_sample_progressive(x_.to(device))
    fig, axes = plt.subplots(1, args.num_diffusion_timesteps + 1, figsize=(10, 8))
    for i in range(args.num_diffusion_timesteps + 1):
        diffuse_data = diffuse_data_pool[i]
        axes[i].hist(diffuse_data, bins=100, range=(-4, 4))
        axes[i].set_title(f"Timestep:{i}")
    plt.show()
    writer.add_figure("Diffusion Data", fig)

    # Training
    losses, time0_bank = train(
        net=net,
        replay_buffer=replay_buffer,
        dataset=myDataset,
        dataloader=dataloader,
        optimizer=optimizer,
        batch_size=args.batch_size,
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        mala_sampler=mala_sampler,
        t_sampler=t_sampler,
        gauss_diffusion=gauss_diffusion,
        logger=logger,
        writer=writer,
        device=device,
        accumulators=accumulators,
        averagemeters=meters,
        dynamic_sampling=args.dynamic_sampling,
        local_jump_enabled=args.local_jump_enabled,
        start_local_epoch=args.start_local_epoch,
        num_diffusion_timesteps=args.num_diffusion_timesteps,
        num_of_points=args.num_of_points,
        num_of_epochs=args.n_epochs,
        n_warm_epochs=args.n_warm_epochs,
        start_reject_epochs=args.start_reject_epochs,
        print_freq=args.print_freq,
    )

    writer.close()

    # Plot loss curve
    fig = plt.figure(figsize=(8, 5))
    plt.plot(losses, label="loss")
    plt.xlabel("n_iter")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    fig.savefig("./figures/daebm-losses.png")

    # Time0 curve
    fig = plt.figure(figsize=(8, 5))
    plt.hist(time0_bank.squeeze(), bins=100, range=(-4, 4), density=True)
    plt.title("Time0 samples")
    plt.legend()
    plt.show()
    fig.savefig("./figures/daebm-time0-samples.png")


if __name__ == "__main__":

    parser = ParserUtils.get_parser(parser_type="daebm")
    args = parser.parse_args()
    ParserUtils.args_check(args, "daebm")
    main(args)
