from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--data_dir', help="Where is the data at?", default='Data/')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--load', action="store_true")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--checkpoint_epochs', type=int, default=1)

    args = parser.parse_args()
    return args
