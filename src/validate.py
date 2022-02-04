from argparse import ArgumentParser
from modules.from_x import validate_from_args


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    validate_from_args(args)


if __name__ == '__main__':
    main()
