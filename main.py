import torch
from lightning.pytorch.cli import LightningCLI, ReduceLROnPlateau

from core.data import *
from core.models import *


class CLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.AdamW)
        parser.add_lr_scheduler_args(ReduceLROnPlateau)


def cli_main():
    cli = CLI(
        NSNET2, 
        NSNET2DataModule, 
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "yaml"}
    )
    

if __name__ == '__main__':
    cli_main()