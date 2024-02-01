""" Preprocessing user input """
import sys
import logging
import pandas as pd

class Configuration:
    def __init__(self, loglevel, csv_file, input_question, output_file):
        self.logger = self.setup_logging(loglevel)
        self.questions = {}
        if csv_file:
            questions_df = pd.read_csv(csv_file[0], sep=";")
            self.questions = questions_df.set_index('id')['question'].to_dict()
        elif input_question:
            self.questions[0] = input_question[0]
        else:
            self.logger.error("It is required to either define an input question as a string or an input csv file.")
            exit(1)

        self.logger.info(f"Loaded questions that shall be answered {self.questions}")

        self.output_file = output_file
        

    def setup_logging(self, loglevel):
        logformat = "[%(asctime)s] %(levelname)s:%(name)s - %(message)s"
        reset = "\x1b[0m"
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
