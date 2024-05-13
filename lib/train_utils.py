import torch
import torch.nn as nn
import re
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
import os

class InitializeNet:
    """ Init parameters of net.

    Methods include:
        kaiming,
        xavier.
    """

    @staticmethod
    def initialize_net(net, net_init_method):

        if net_init_method.startswith("kaiming"):
            if (
                "[" in net_init_method
                and "out" in re.findall(r"\[(.*?)\]", net_init_method)[0]
            ):
                net = InitializeNet.kaiming_init_net(net, net.act_func, mode="fan_out")
            else:
                net = InitializeNet.kaiming_init_net(net, net.act_func)
        elif net_init_method.startswith("xavier"):
            init_std = (
                float(re.findall(r"[-+]?\d*\.\d+|\d+", net_init_method)[0])
                if "[" in net_init_method
                else 1
            )
            net = InitializeNet.xavier_init_net(net, net.act_func, init_std)
        elif net_init_method == "default":
            pass
        else:
            raise NotImplementedError

        return net

    def kaiming_init_net(net, act_func, mode="fan_in"):
        "act_funct: input an object of activation function"
        negative_slope = 0
        if isinstance(act_func, torch.nn.modules.activation.ReLU):
            nonlinearity = "relu"
        elif isinstance(act_func, torch.nn.modules.activation.LeakyReLU):
            negative_slope = act_func.negative_slope
            nonlinearity = "leaky_relu"
        else:
            nonlinearity = "linear"

        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, a=negative_slope, mode=mode, nonlinearity=nonlinearity
                )
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(
                    m.weight, a=negative_slope, mode=mode, nonlinearity=nonlinearity
                )
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        return net

    def xavier_init_net(net, act_func, std=1):
        "act_funct: input an object of activation function"
        if isinstance(act_func, torch.nn.modules.activation.ReLU):
            gain = nn.init.calculate_gain("relu")
        elif isinstance(act_func, torch.nn.modules.activation.LeakyReLU):
            gain = nn.init.calculate_gain("leaky_relu", act_func.negative_slope)
        else:
            gain = nn.init.calculate_gain("linear")

        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain)
                m.weight.data *= std
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain)
                m.weight.data *= std
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        return net

class OptimizerConfigure:
    """ Configure optimizer of EBM training or DAEBM"""

    @staticmethod
    def configure_optimizer(
        params, optimizer_type, lr, weight_decay=0, betas=[0.9, 0.999], sgd_momentum=0.9
    ):

        if optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                params, lr=lr, betas=betas, weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=lr, weight_decay=weight_decay, momentum=sgd_momentum
            )
        else:
            raise NotImplementedError("Wrong optimizer type")

        return optimizer

class SchedulerConfigure:
    """Configure scheduler and warmup scheduler"""

    @staticmethod
    def configure_scheduler(scheduler_type, optimizer, **kwargs) -> _LRScheduler:
        """Configure scheduler and warm_shcheduler according to different type

        Using match-case, which is supported in python 3.10 or above.

        if scheduler_type == "MultiStep":
            kwargs:{
                milestones (list)
                lr_decay_factor (float)
            }
        elif scheduler_type == "Linear":
            kwargs:{
                milestones (list)
                lr_decay_factor (float)
                n_epochs
            }
        elif scheduler_type == "LinearPlateau":
            kwargs:{
                milestones (list)
                lr_decay_factor (float)
            }
        else
            raise Error
        """
        match scheduler_type:
            case "MultiStep":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, kwargs["milestones"], gamma=kwargs["lr_decay_factor"]
                )
            case "Linear":
                linear_decay_slope = (1 - kwargs["lr_decay_factor"]) / (
                    kwargs["n_epochs"] - kwargs["milestones"][0] + 1
                )
                lambda_lr_func = lambda epoch: (
                    1 - linear_decay_slope * (epoch - kwargs["milestones"][0] + 1)
                    if epoch >= kwargs["milestones"][0]
                    else 1
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda_lr_func
                )
            case "LinearPlateau":
                linear_decay_slope = (1 - kwargs["lr_decay_factor"]) / (
                    kwargs["milestones"][1] - kwargs["milestones"][0] + 1
                )
                lambda_lr_func = lambda epoch: (
                    1 - linear_decay_slope * (epoch - kwargs["milestones"][0] + 1)
                    if epoch >= kwargs["milestones"][0]
                    and epoch < kwargs["milestones"][1]
                    else (
                        kwargs["lr_decay_factor"]
                        if epoch >= kwargs["milestones"][1]
                        else 1
                    )
                )  # noqa
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda_lr_func
                )
            case _:
                raise NotImplementedError

        return scheduler
    
    class WarmUpLR(_LRScheduler):
            """WarmUp training learning rate scheduler."""

            def __init__(
                self, optimizer: Optimizer, total_iters: int, last_epoch: int = -1
            ) -> None:
                self.total_iters = total_iters
                super().__init__(optimizer, last_epoch)

            def get_lr(self):
                """Return a list of learning rate of different parameters.

                For batch m, return base_lr * m / total_iters
                """
                return [
                    base_lr * self.last_epoch / (self.total_iters + 1e-8)
                    for base_lr in self.base_lrs
                ]

    @staticmethod
    def configure_warmup_scheduler(optimizer, n_warm_iters) -> WarmUpLR:
        return SchedulerConfigure.WarmUpLR(optimizer, n_warm_iters)

def save_checkpoint(state, save_path_prefix="", filename="daebm_checkpoint.pt"):
    if not os.path.exists(os.path.dirname(save_path_prefix)):
        os.makedirs(os.path.dirname(save_path_prefix))
    torch.save(state, save_path_prefix + filename)

