import torch

#from dataloader import ImageData
from parse import get_args
import torchvision
from torchvision import transforms

from training import run_model

args = get_args()

#dataset = ImageData(args.data_dir)
dataset = torchvision.datasets.MNIST("D:/Documents/MasterRCI/WS20_EPFL/SemesterProject/VAE", train=False, download=True, transform=transforms.ToTensor())


if args.mode == 'train':
    dataset = torchvision.datasets.MNIST("D:/Documents/MasterRCI/WS20_EPFL/SemesterProject/VAE", train=True, download=True, transform=transforms.ToTensor())

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

elif args.mode == 'test':
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

elif args.mode == 'inference':
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

run_model(dataloader, args=args)