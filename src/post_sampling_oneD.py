# Post sampling for one dimesion Gaussian experient
import torch
import logging
import matplotlib.pyplot as plt
import os
import sys
import shutil

cwd = os.getcwd()
sys.path.append(cwd)

try:
    from lib.model import MyUfuncTemb
    from lib.diffusion import GaussianDiffusion
    from lib.sampler import MGMS_sampling
    from lib.create_dataset import GaussFourDataset
    from lib.config_parser import ParserUtils

except ImportError:
    raise

def main(args):
    
    torch.manual_seed(args.t_seed)
    logging.basicConfig(
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger("DAEBM (Post Sampling)")
    net = MyUfuncTemb(
        in_ch=1, n_timesteps=args.num_diffusion_timesteps+1,act_func=args.act_func
    )

    exp_dir = os.path.join(args.main_dir, args.pexp_dir, args.exp_dir)
    pexp_dir = os.path.join(args.main_dir, args.pexp_dir)
    checkpoint = torch.load(
        os.path.join(pexp_dir, "saved_models/net_checkpoint.pt")
    ) # epoch, state_dict, step_size
    net.load_state_dict(checkpoint["state_dict"])
    
    init_step_size = checkpoint["step_size"]
    print(init_step_size)

    # Diffusion configure
    gauss_diffusion = GaussianDiffusion(
        n_timesteps=args.num_diffusion_timesteps,
        beta_schedule=args.diffusion_schedule,
        beta_start=args.diffusion_betas[0],
        beta_end=args.diffusion_betas[1]
    )

    # MGMS configure
    # May have cleaner form, but below is easy for us to understand
    is_reject = (
        True if (args.mala_isreject is True or args.dynamic_sampling is True) else False
    )

    # Device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    save_figures_dir = os.path.join(exp_dir, "figures")
    try:
        os.makedirs(save_figures_dir)
    except:
        shutil.rmtree(save_figures_dir)
        os.makedirs(save_figures_dir)

    mala_sampler = MGMS_sampling(
        num_steps=args.sample_steps,
        init_step_size=init_step_size,
        is_reject=is_reject,
        device=device
    )

    sampling_chains = args.sampling_chains
    # Init by noise
    init_x_t_neg = torch.randn((sampling_chains,1),device=device)
    init_t_neg = torch.ones(sampling_chains,device=device).fill_(args.num_diffusion_timesteps).unsqueeze(-1).long()

    count0=torch.zeros_like(init_t_neg,device="cpu")
    mark0=torch.zeros_like(init_t_neg,device="cpu")
    time0_bank = torch.randn((0,1))

    for n_iter in range(args.total_iters):

        # MGMS sampling
        x_t_neg, t_neg, acpt_rate = mala_sampler.mgms_sampling(
            net,
            init_x_t_neg,
            init_t_neg
        )

        count0 += (t_neg==0)
        time0_samples_idx = torch.logical_and(count0 == args.stop_a_chain_M, mark0 == 0)
        mark0[time0_samples_idx] = torch.ones_like(mark0[time0_samples_idx]) 

        time0_samples = x_t_neg[time0_samples_idx].unsqueeze(-1)
        time0_bank = torch.cat([time0_bank, time0_samples], 0)

        init_x_t_neg, init_t_neg = x_t_neg.to(device), t_neg.to(device)
        if n_iter % 50 ==0:
            logger.info(f"n_iter:{n_iter}, size of time0_bank is {time0_bank.shape}")
    
    myDataset = GaussFourDataset(num_of_points=args.num_of_points)
    import pdb
    pdb.set_trace()
    fig = plt.figure()
    x= torch.linspace(-4,4,100).unsqueeze(-1)
    plt.hist(time0_bank.squeeze(),bins=100,range=(-4,4),density=True)
    plt.plot(x.squeeze(),myDataset.pdf0(x).squeeze(),label="True density")
    plt.title("Results of Post-Sampling (DAEBM)")
    fig.savefig(os.path.join(save_figures_dir,"daebm-postsampling.png"))

    # Long run sampling: fron real data
    mgms_sampler = MGMS_sampling(
        num_steps=1000,
        init_step_size=init_step_size,
        is_reject=is_reject,
        device=device
    )
    t = torch.zeros((args.num_of_points,1),device=device).long()
    long_run_samples, long_run_labels, _ = mgms_sampler.mgms_sampling(
        net,
        myDataset.get_full_data().unsqueeze(-1).to(device),
        t
    )
    fig = plt.figure()
    time0_long_run_samples = long_run_samples[long_run_labels==0]
    plt.hist(time0_long_run_samples.squeeze(),bins=100,range=(-4,4))
    plt.title("Results of Long-run-Sampling (DAEBM)")
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(save_figures_dir,"daebm-longrunsampling.png"))

    
if __name__ =="__main__":

    parser = ParserUtils.get_parser(parser_type='postsampling')
    args = parser.parse_args()
    ParserUtils.args_check(args=args, parser_type='postsampling') # 同样未完成
    main(args)
    





