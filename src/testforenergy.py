# 测试能量是否相等
# 针对从四模态高斯分布（1000点）抽出100个点后，与从replay_buffer（标准正态分布）抽出
# 100个点计算能量，看是否相等
import torch
import os
from oneD_Gaussian_DAEBM import MyUfuncTemb,GaussianDiffusion,MGMS_sampling
import logging
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt
import shutil

def main(args):
    
    torch.manual_seed(args.t_seed)
    logging.basicConfig(
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger("DAEBM (Post Sampling)")

    checkpoint = torch.load(
        "./results/daebm_checkpoint.pt"
    ) # epoch, state_dict, step_size

    net = MyUfuncTemb(
        in_ch=1, n_timesteps=args.num_diffusion_timesteps+1,act_func=args.act_func
    )
    net.load_state_dict(checkpoint["state_dict"])
    init_step_size = checkpoint["step_size"]

    gauss_diffusion = GaussianDiffusion(
        n_timesteps=args.num_diffusion_timesteps,
        beta_schedule=args.diffusion_schedule,
        beta_start=args.diffusion_betas[0],
        beta_end=args.diffusion_betas[1],
    )

    mala_sampler = MGMS_sampling(
        num_steps=args.sample_steps,
        init_step_size=init_step_size,
        is_reject=args.mala_isreject,
    )
    log_dir=args.log_dir
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    dataset = GaussFourDataset(args.num_of_points)

    replay_buffer = torch.load("./results/replay_buffer.pt")
    replay_buffer_data = replay_buffer["samples"]
    replay_buffer_label = replay_buffer["labels"]
    
    logger.info(f"size of replay_buffer of time 0:{(replay_buffer_label==0).sum()}")
    # 从dataset中随机抽取batch_size个元素
    
    for i in range(100):
        perm_indices = torch.randperm(args.num_of_points)
        selected_indices = perm_indices[:args.batch_size]
        select_data = dataset.data[selected_indices].unsqueeze(-1)
        t = torch.randint(high=args.num_diffusion_timesteps+1,size=(args.batch_size,1)).long()
        
        x_t = gauss_diffusion.q_sample(select_data,t)
        # import pdb
        # pdb.set_trace()

        init_x_t_neg, init_t_neg = replay_buffer_data[selected_indices], replay_buffer_label[selected_indices].unsqueeze(-1)

        x_t_neg, t_neg, _ = mala_sampler.mgms_sampling(net, init_x_t_neg, init_t_neg)
        
        writer.add_histogram("Train set/x_t",x_t, i)

        fig = plt.figure()
        plt.hist(t.detach().squeeze(),bins=range(0,args.num_diffusion_timesteps+2))

        writer.add_figure("Train set/t", fig, i)
        writer.add_histogram("Replay buffer/x_neg", x_t_neg, i)

        fig = plt.figure()
        plt.hist(t_neg.detach().squeeze(),bins=range(0,args.num_diffusion_timesteps+2))
        writer.add_figure("Replay buffer/t_neg",fig, i)
        energy = net.energy_output(x_t, t)
        energy_mean_pos = energy.mean().detach()

        energy_neg = net.energy_output(x_t_neg, t_neg)
        energy_mean_neg = energy_neg.mean().detach()
        writer.add_scalars(
            main_tag="Energy",
            tag_scalar_dict={
                "pos energy":energy_mean_pos,
                "neg energy":energy_mean_neg,
                "total loss":energy_mean_pos-energy_mean_neg
            },
            global_step=i
        )

    
    # 取出time 0
    time0_data = replay_buffer_data[replay_buffer_label==0]
    time0_data = time0_data[:100]
    fig = plt.figure()
    plt.hist(time0_data.squeeze(),bins=100,range=(-4,4))
    writer.add_figure("Replay buffer time0 sample", fig, 0)

    perm_indices = torch.randperm(args.num_of_points)
    selected_indices = perm_indices[:100]
    select_data = dataset.data[selected_indices].unsqueeze(-1)
    fig = plt.figure()
    plt.hist(select_data.squeeze(),bins=100,range=(-4,4))
    writer.add_figure("Dataset sample", fig, 0)
    
    energy_neg = net.energy_output(time0_data)
    energy_pos = net.energy_output(select_data)
    logger.info(f"pos energy:{energy_pos.mean().item()}, neg energy:{energy_neg.mean().item()}")
    # for name, param in net.named_parameters():
    #     writer.add_histogram(name, param, global_step=0)


if __name__=="__main__":
    parser = ParserUtils.get_parser("postsampling")
    args = parser.parse_args()
    main(args)
