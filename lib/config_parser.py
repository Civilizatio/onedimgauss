import argparse
from typing import Literal


class ParserUtils:
    """ Get different parsers.

    Types include:
        | daebm | ebm | postsampling |
    """

    @staticmethod
    def get_parser(parser_type: Literal["daebm", "ebm", "postsampling"] = "daebm"):
        match parser_type:
            case "ebm":
                parser = ParserUtils._get_ebm_parser()
            case "daebm":
                parser = ParserUtils._get_daebm_parser()
            case "postsampling":
                parser = ParserUtils._get_post_sampling_parser()
            case _:
                raise NotImplementedError
        return parser

    @staticmethod
    def args_check(
        args, parser_type: Literal["daebm", "ebm", "postsampling"] = "daebm"
    ):
        """Check validation of args."""
        match parser_type:
            case "ebm":
                return ParserUtils._args_check_for_ebm(args)
            case "daebm":
                return ParserUtils._args_check_for_daebm(args)
            case "postsampling":
                return ParserUtils._args_check_for_postsampling(args)
            case _:
                raise NotImplementedError

    def _args_check_for_daebm(args):
        """Check for validation of daebm

        Belows are rules:

        For Langevin Dynamics:
            1. `start_local_epoch` should be less than `n_epochs` when
                `local_jump_enabled` is True.
            2. `window_size` should be an odd
            3. length of means vector should be the same as that of peak_ratio
            4. `sample_method_of_t` must be one of normal, uniform, exponential.

        """

        if args.local_jump_enabled and args.start_local_epoch > args.n_epochs:
            raise argparse.ArgumentError(
                None,
                "When using local jump, start local jump should be less than n_epochs",
            )

        if args.window_size % 2 == 0:
            raise argparse.ArgumentError(
                None, "Window size of local neighbor should be an odd"
            )

        if len(args.means) != len(args.peaks_ratio):
            raise argparse.ArgumentError(
                None, "length of means should be the same as the peaks_ratio"
            )

        if not args.sample_method_of_t in ["uniform", "normal", "exponential"]:
            raise argparse.ArgumentError(
                None,
                "sample methods of t must be one of uniform, normal and exponential",
            )

    def _args_check_for_ebm(args):
        pass

    def _args_check_for_postsampling(args):
        pass

    def check_none_or_int(x):
        if x == "None":
            return None
        else:
            try:
                x = int(x)
            except ValueError:
                raise argparse.ArgumentTypeError("%r not a integer literal" % (x,))
            return x

    def _get_ebm_parser():
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--log_dir",
            type=str,
            default="tb_log/",
            help="path and prefix of tensorboard log file",
        )
        parser.add_argument(
            "--main_dir",
            type=str,
            required=True,
            help="directory of the experiment: specify as ./ebm, ./daebm"
        )
        parser.add_argument(
            "--exp_dir",
            type=str,
            default="experiments/"
        )
        # Dataset Configure
        parser.add_argument(
            "-n",
            "--num_of_points",
            type=int,
            default=1000,
            help="num of points of Gaussian data set",
        )

        # Sampler Configure
        parser.add_argument(
            "--mala_isreject",
            action="store_true",
            default=False,
            help="whether using reject when langevin sampling",
        )

        parser.add_argument(
            "--mala_sigma", type=float, default=0.1, help="sigma of langevin dynamics"
        )

        parser.add_argument(
            "--mala_steps", type=int, default=10, help="steps of langevin transposition"
        )

        # Network Configure
        parser.add_argument(
            "--act_func", type=str, default="Softplus", help="activation function"
        )

        # Learning Parameters
        parser.add_argument("--batch_size", type=int, default=100, help="batch size")

        parser.add_argument("--n_iter", type=int, default=1200, help="num of iters")

        parser.add_argument("--lr", type=float, default=0.2, help="learning rate")

        parser.add_argument(
            "--weight_decay", type=float, default=0, help="weight decay of optimizer"
        )

        parser.add_argument(
            "--milestones",
            nargs="+",
            type=int,
            default=[40, 60, 80, 100],
            help="scheduler adjust milestones",
        )

        parser.add_argument(
            "--lr_decay_factor",
            type=float,
            default=0.2,
            help="learning rate decay factor",
        )

        return parser

    def _get_daebm_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--t_seed", type=int, default=2024, help="random seed")
        parser.add_argument(
            "--print_freq",
            type=int,
            default=100,
            help="number of iterations for printing log during training",
        )
        parser.add_argument(
            "--log_dir",
            type=str,
            default="tb_log/",
            help="path and prefix of tensorboard log file",
        )
        parser.add_argument(
            "--main_dir",
            type=str,
            required=True,
            help="directory of the experiment: specify as ./ebm, ./daebm"
        )
        parser.add_argument(
            "--exp_dir",
            type=str,
            default="experiments/"
        )

        parser.add_argument("--saved_models_dir", type=str,
                        default="saved_models/net", help="prefix of saved models")
        # Dataset Parameters
        parser.add_argument(
            "-n",
            "--num_of_points",
            type=int,
            default=1000,
            help="num of points of Gaussian data set",
        )

        parser.add_argument(
            "--means", type=float, nargs="+", help="means of Gauss kernels"
        )
        parser.add_argument(
            "--peaks_ratio", type=float, nargs="+", help="peaks ratio of Gauss kernels"
        )

        # Learning Parameters
        parser.add_argument(
            "--n_epochs", type=int, default=200, help="number of epochs to training"
        )

        parser.add_argument(
            "--n_warm_epochs", type=int, default=10, help="number of epochs to training"
        )

        parser.add_argument("--batch_size", type=int, default=100, help="batch size")
        parser.add_argument(
            "--start_reject_epochs", type=int, default=None, help="start_reject_epochs"
        )
        # Network Parameters
        parser.add_argument(
            "--act_func", type=str, default="Softplus", help="activation function"
        )
        parser.add_argument(
            "--time_embedding_type",
            type=str,
            default="sin",
            help="time embedding methods, sin or onehot",
        )
        parser.add_argument(
            "--net_init_method",
            dest="net_init_method",
            type=str,
            default="kaiming",
            help="network initialization method",
        )
        parser.add_argument("--cuda", type=int, default="0", help="device-name")
        # Sampling Parameters
        parser.add_argument(
            "--mala_isreject",
            action="store_true",
            default=False,
            help="whether using reject when langevin sampling",
        )
        parser.add_argument(
            "--sample_steps", type=int, default=30, help="sampling steps"
        )
        parser.add_argument(
            "--replay_buffer_size",
            type=ParserUtils.check_none_or_int,
            default=None,
            help="replay buffer size, if not speicified, we run short-run",
        )
        parser.add_argument(
            "--random_type",
            type=str,
            default="normal",
            help="random images type, uniform, normal",
        )
        parser.add_argument(
            "--b_factor",
            type=float,
            default=2e-4,
            help="step-size factor in Langevin sampling",
        )
        parser.add_argument(
            "--dynamic_sampling",
            action="store_true",
            default=False,
            help="dynamically adjust sampling steps after warm-up iterations",
        )

        parser.add_argument(
            "--local_jump_enabled",
            action="store_true",
            default=False,
            help="Whether to use local jump",
        )

        parser.add_argument(
            "--start_local_epoch",
            type=int,
            default=5,
            help="Which epoch to start local jump instead of global jump",
        )

        parser.add_argument(
            "--window_size", type=int, default=5, help="size of neighbors in Local jump"
        )
        # T_sampler configure
        parser.add_argument(
            "--sample_method_of_t",
            type=str,
            default="uniform",
            help="must be one of uniform, exponential and normal",
        )
        
        # Optimizer Parameters
        parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        parser.add_argument(
            "--optimizer_type",
            dest="optimizer_type",
            type=str,
            default="adam",
            help="adam or sgd",
        )

        parser.add_argument(
            "--betas",
            nargs=2,
            metavar=("beta1", "beta2"),
            default=(0.9, 0.999),
            type=float,
            help="beta parameters for adam optimizer",
        )

        parser.add_argument(
            "--sgd_momentum", type=float, default=0.0, help="momentum in sgd"
        )

        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.0,
            help="weight decay for discriminator",
        )

        parser.add_argument(
            "--scheduler_type",
            type=str,
            default="LinearPlateau",
            help="learning rate scheduler: MultiStep, Linear, LinearPlateau",
        )

        parser.add_argument(
            "--lr_decay_factor",
            type=float,
            default=1e-4,
            help="learning rate decay factor",
        )

        parser.add_argument(
            "--milestones",
            nargs="+",
            type=int,
            default=[120, 180],
            help="milestones of learning rate decay",
        )

        # Diffusion Parameters
        parser.add_argument(
            "--num_diffusion_timesteps",
            type=int,
            default=6,
            help="num of diffusion steps",
        )
        parser.add_argument(
            "--diffusion_schedule",
            type=str,
            default="linear",
            help="type of diffusion schedule: linear, sigmoid, quad, sqrtcumlinear",
        )
        parser.add_argument(
            "--diffusion_betas",
            nargs=2,
            metavar=("beta1", "beta2"),
            default=(1e-5, 5e-3),
            type=float,
            help="starting and ending betas of diffusion scheduler",
        )

        return parser

    def _get_post_sampling_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--t_seed", type=int, default=2024, help="random seed")
        parser.add_argument(
            "--act_func", type=str, default="Softplus", help="activation function"
        )
        parser.add_argument("--batch_size", type=int, default=100, help="batch size")
        parser.add_argument(
            "--main_dir",
            type=str,
            required=True,
            help="directory of the experiment: specify as ./ebm, ./daebm"
        )
        parser.add_argument(
            "--exp_dir",
            type=str,
            default="experiments/"
        )
        parser.add_argument("--saved_models_dir", type=str,
                        default="saved_models/net", help="prefix of saved models")
        parser.add_argument(
            "--log_dir",
            type=str,
            default="tb_log/",
            help="path and prefix of tensorboard log file",
        )
        # Diffusion Parameters
        parser.add_argument(
            "--num_diffusion_timesteps",
            type=int,
            default=6,
            help="num of diffusion steps",
        )
        parser.add_argument(
            "--diffusion_schedule",
            type=str,
            default="linear",
            help="type of diffusion schedule: linear, sigmoid, quad, sqrtcumlinear",
        )
        parser.add_argument(
            "--diffusion_betas",
            nargs=2,
            metavar=("beta1", "beta2"),
            default=(1e-5, 5e-3),
            type=float,
            help="starting and ending betas of diffusion scheduler",
        )

        # Samping configure
        parser.add_argument(
            "--mala_isreject",
            action="store_true",
            default=False,
            help="whether using reject when langevin sampling",
        )
        parser.add_argument(
            "--sample_steps", type=int, default=30, help="sampling steps"
        )
        parser.add_argument(
            "--dynamic_sampling",
            action="store_true",
            default=False,
            help="dynamically adjust sampling steps after warm-up iterations",
        )

        parser.add_argument(
            "--sampling_chains", type=int, default=100, help="num of parallel sampling"
        )

        parser.add_argument(
            "--total_iters", type=int, default=500, help="total iters of post sampling"
        )
        parser.add_argument(
            "--stop_a_chain_M",
            type=int,
            default=40,
            help="stop sampling until M times ",
        )
        parser.add_argument(
            "-n",
            "--num_of_points",
            type=int,
            default=1000,
            help="num of points of Gaussian data set",
        )
        return parser
