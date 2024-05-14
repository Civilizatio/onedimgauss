# DAEBM中一维高斯实验的复现

import torch
from torch.utils.data import DataLoader
import os
import sys
import logging
import argparse
import matplotlib.pyplot as plt
import shutil
from torch.utils.tensorboard import SummaryWriter

cwd = os.getcwd()
sys.path.append(cwd)

try:
    from lib.config_parser import ParserUtils
    from lib.create_dataset import GaussFourDataset
    from lib.model import MyUfuncOneD, MyUfunc
    from lib.sampler import MALA_Sampling
    from lib.replay_buffer import ReplayBufferEBM
    from lib.train_utils import (
        OptimizerConfigure,
        SchedulerConfigure,
        save_checkpoint,
    )
    from lib.plot_utils import (
        plot_results_of_ebm,
        plot_energy_function
    )
except ImportError:
    raise

def get_data_loader(dataset, batch_size=100):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return dataloader


def training_losses(net, x_pos, x_neg):
    """ loss = 1/N \sum U(x_{p_data})-U(x_{p_theta})"""

    pos_energy = net(x_pos).squeeze().mean()
    neg_energy = net(x_neg).squeeze().mean()
    loss = (pos_energy - neg_energy)

    return loss, pos_energy, neg_energy


def train(
    net:MyUfuncOneD|MyUfunc,
    replay_buffer:ReplayBufferEBM,
    dataset:GaussFourDataset,
    data_loader:DataLoader,
    optimizer:torch.optim.Optimizer,
    batch_size:int,
    scheduler:torch.optim.lr_scheduler.MultiStepLR,
    mala_sampler:MALA_Sampling,
    logger:logging.Logger,
    writer:SummaryWriter,
    saved_models_dir:str,
    num_of_points:int=1000,
    num_of_iters:int=1200,
):
    losses = []
    num_of_epochs = num_of_iters // (num_of_points // batch_size)
    niter = 0

    for epoch in range(num_of_epochs):
        for _ in range(num_of_points // batch_size):
            niter += 1

            x_pos = next(iter(data_loader)).unsqueeze(-1)

            x_neg_init, buffer_idx = replay_buffer.sample_from_buffer(batch_size)
            x_neg = mala_sampler.mala_sampling(net, x_neg_init)

            # calculate loss
            loss, pos_energy, neg_energy = training_losses(net, x_pos=x_pos, x_neg=x_neg)

            if torch.isnan(loss) or loss.abs().item() > 1e8:
                logger.error(f"Training breakdown, loss is {loss}")

            # parameters update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            # Acquire norm of grads of loss
            grads_norms = torch.tensor([param.grad.norm().item() for param in net.parameters()])
            grad_norm_sum = torch.norm(grads_norms)

            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={
                    "loss":loss.item(),
                    "pos_energy":pos_energy.item(),
                    "neg_energy":neg_energy.item()
                },
                global_step=niter
            )

            writer.add_scalar("Loss/loss_grad_norm", grad_norm_sum.item(),global_step=niter)
            # update replay buffer
            replay_buffer.update_buffer(buffer_idx, x_neg)

            # Plot replay buffer figures
            fig = plt.figure()
            x_tmp = torch.linspace(-4, 4, 100).unsqueeze(-1)
            y, _ = replay_buffer.sample_from_buffer(num_of_points)
            y = y.squeeze()
            y_truth = dataset.pdf0(x_tmp.clone().detach()).squeeze()
            plt.hist(y, bins=100, range=(-4, 4), density=True)
            plt.plot(x_tmp, y_truth, color="red", label="true density")
            plt.title(f"Replay buffer iter={niter}")
            plt.legend()
            writer.add_figure("Replay buffer",fig, global_step=niter)


        else:
            scheduler.step()
            if niter % 100 == 0:
                logger.info(f"n_iter:{niter}, loss:{loss}")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                    },
                    save_path_prefix=saved_models_dir,
                )
                
            if niter % 200 == 0:
                writer.add_figure("Energy function",plot_energy_function(dataset=dataset, net=net),global_step=niter)
                
    return losses

def main(args):

    # Logger configure
    logging.basicConfig(
        format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    logger = logging.getLogger("EBM Training")

    # Dir config
    exp_dir = os.path.join(args.main_dir, args.exp_dir)
    try:
        os.makedirs(exp_dir)
    except FileExistsError:
        shutil.rmtree(exp_dir)
        os.makedirs(exp_dir)

    # dir for saving models
    saved_models_dir = os.path.join(exp_dir, args.saved_models_dir)
    
    # dir for saving figures
    save_figures_dir = os.path.join(exp_dir, "figures")
    os.makedirs(save_figures_dir)

    # Tensorboard configure
    log_dir = os.path.join(exp_dir,args.log_dir)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)


    # Parameters configure
    torch.manual_seed(2024)
    net = MyUfuncOneD(act_func=args.act_func)
    dataset = GaussFourDataset(num_of_points=args.num_of_points)
    dataloader = get_data_loader(dataset=dataset, batch_size=args.batch_size)
    optimizer = OptimizerConfigure.configure_optimizer(
        params=net.parameters(), optimizer_type="sgd",lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = SchedulerConfigure.configure_scheduler(
        scheduler_type="MultiStep",
        optimizer=optimizer,
        milestones=args.milestones,
        lr_decay_factor=args.lr_decay_factor,
    )

    replay_buffer = ReplayBufferEBM(args.num_of_points)
    mala_sampler = MALA_Sampling(
        is_reject=args.mala_isreject, mala_sigma=args.mala_sigma, mala_n=args.mala_steps
    )

    losses = train(
        net=net,
        replay_buffer=replay_buffer,
        dataset=dataset,
        data_loader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=args.batch_size,
        mala_sampler=mala_sampler,
        logger=logger,
        writer=writer,
        saved_models_dir=saved_models_dir,
        num_of_points=args.num_of_points,
        num_of_iters=args.n_iter,
    )

    # plot loss curve
    fig = plt.figure(figsize=(8, 5))
    plt.plot(losses, label="loss")
    plt.xlabel("n_iter")
    plt.ylabel("loss")
    plt.legend()
    fig.savefig(os.path.join(save_figures_dir,"ebm_losses.png"))

    # Plot results
    fig = plot_results_of_ebm(dataset=dataset,net=net,replay_buffer=replay_buffer)
    writer.add_figure("Results",fig,global_step=0)
    fig.savefig(os.path.join(save_figures_dir,"sampling_results.png"))

    # distribution
    x = torch.linspace(-4, 4, 100).unsqueeze(-1)
    energy_truth = dataset.ufunc0(x).squeeze()
    energy_model = net(x).detach().squeeze()

    energy_truth_normalized = energy_truth - torch.min(energy_truth)
    energy_model_normalized = energy_model - torch.min(energy_model)

    density_truth_normalized = dataset.pdf0(x).squeeze()
    density_model_normalized = torch.exp(-energy_model_normalized)

    fig = plt.figure(figsize=(8,5))
    plt.plot(
        x.squeeze(),
        density_truth_normalized,
        linewidth=2,
        markersize=12,
        label="True",
        color="#ff7f0e",
    )
    plt.plot(
        x.squeeze(),
        density_model_normalized,
        linewidth=2, markersize=12, label="Learned"
    )
    fig.savefig(os.path.join(save_figures_dir,"ebm_distributions.png"))


if __name__ == "__main__":
    
    parser = ParserUtils.get_parser(parser_type="ebm")
    args = parser.parse_args()
    ParserUtils.args_check(args=args, parser_type="ebm") # 这里还没写
    main(args)
    