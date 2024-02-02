import argparse
import sys


from .configuration import Configuration


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Adversarial machine learning robustness evaluations for ML based N-IDS."
    )
    parser.add_argument(
        "-c",
        "--csv",
        metavar="file",
        type=str,
        required=False,
        dest="csv_file",
        nargs=1,
        help="relative path to the csv file configuration the questions to be answered. If not defined, it is required to define an input question as a string.",
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="question",
        type=str,
        dest="input_question",
        nargs=1,
        help="String input question to be answered. If not defined, it is required to define an input csv file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="file",
        type=str,
        dest="output_file",
        nargs=1,
        help="relative path to the file in which the output shall be written to. If not defined, the prediction results will be displayed on the standard output.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        dest="loglevel",
        const="INFO",
        action="store_const",
        help="Set logging level to INFO to obtain information about the internal program execution. Without the verbose flag, only Errors are diplayed in stdout.",
    )
    return parser.parse_args(args)


def cli(args):
    args = parse_args(args)
    config = Configuration(
        args.loglevel, args.csv_file, args.input_question, args.output_file
    )
    print(config.get_next_element())


def main():
    cli(sys.argv[1:])
