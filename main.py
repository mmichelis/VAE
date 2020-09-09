# ------------------------------------------------------------------------------
# This file should be called using command line, the command is then parsed and 
# finds the correct files to run.
# ------------------------------------------------------------------------------


import torch
import torchvision

from torchvision import transforms

from parse import get_args
from training import run_model


if __name__ == "__main__":
    args = get_args()

    dataset = torchvision.datasets.MNIST("Data", train=False, download=True, transform=transforms.ToTensor())


    if args.mode == 'train':
        dataset = torchvision.datasets.MNIST("Data", train=True, download=True, transform=transforms.ToTensor())

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    elif args.mode == 'test' or args.mode == 'inference':
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )


    run_model(dataloader, args=args)