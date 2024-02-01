""" Preprocessing user input """
import sys
import logging

class Configuration:
    def __init__(self, loglevel, csv_file, input_question, output_file):
        self.logger = self.setup_logging(loglevel)
        if not csv_file and not input_question:
            self.logger.error("It is required to either define an input question as a string or an input csv file.")
            exit(1)

    def setup_logging(self, loglevel):
        logformat = "[%(asctime)s] %(levelname)s:%(name)s - %(message)s"
        reset = "\x1b[0m"
        # TODO: Set color scheme for each logging level, Error and info
        FORMATS = {
            logging.ERROR: "\x1b[31;20m" + logformat + reset,
            logging.INFO: "\x1b[38;20m" + logformat + reset,
        }
        logging.basicConfig(
            level=loglevel,
            stream=sys.stdout,
            format=logformat,
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
        return logging.getLogger(__name__)
