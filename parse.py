# ------------------------------------------------------------------------------
# Parser function for command line arguments.
# ------------------------------------------------------------------------------

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'test', 'inference'], default='train')
    parser.add_argument('--load', help="Loads existing trained model if available.", action="store_true")

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', help="Learning to start training with.", type=float, default=5e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--checkpoint_epochs', help="Number of epochs between two checkpoints.", type=int, default=3)

    args = parser.parse_args()
    return args
