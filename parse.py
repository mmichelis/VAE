# ------------------------------------------------------------------------------
# Parser function for command line arguments.
# ------------------------------------------------------------------------------

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'test', 'inference'], default='train')
    parser.add_argument('--load', help="Loads existing trained model if available.", action="store_true")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--checkpoint_epochs', help="Number of epochs between two checkpoints.", type=int, default=1)

    args = parser.parse_args()
    return args
