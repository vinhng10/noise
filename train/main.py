import torch
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.cli import LightningCLI, ReduceLROnPlateau

from data import *
from models import *
from schedulers import *


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.AdamW)
        # parser.add_optimizer_args(torch.optim.AdamW, nested_key="discriminator_optimizer")
        parser.add_lr_scheduler_args(ReduceLROnPlateau)


def cli_main():
    cli = CLI(
        VADLightningMobileNetV1,
        VADNoiseDataModule,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "yaml"},
    )


if __name__ == "__main__":
    cli_main()
