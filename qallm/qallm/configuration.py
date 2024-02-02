""" Preprocessing user input """
from lingua import Language, LanguageDetectorBuilder
import sys
import logging
import pandas as pd


class Configuration:
    def __init__(self, loglevel, csv_file, input_question, output_file):
        self.logger = self.setup_logging(loglevel)
        self.questions = {}
        if csv_file:
            questions_df = pd.read_csv(csv_file[0], sep=";")
            self.questions = questions_df.set_index("id")["question"].to_dict()
        elif input_question:
            self.questions[0] = input_question[0]
        else:
            self.logger.error(
                "It is required to either define an input question as a string or an input csv file."
            )
            exit(1)

        self.logger.info(f"Loaded questions that shall be answered {self.questions}")

        # Initialise language detection model
        self.language_detector = self._initialise_language_detector()

        self.output_file = output_file

    def setup_logging(self, loglevel):
        logformat = "[%(asctime)s] %(levelname)s:%(name)s - %(message)s"
        logging.basicConfig(
            level=loglevel,
            stream=sys.stdout,
            format=logformat,
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
        return logging.getLogger(__name__)

    def get_next_element(self):
        # get the next question to handle
        question_index = next(iter(self.questions))
        question = self.questions[question_index]
        question_language = self._find_language(question)
        # return the id, question, and the language in which the question was asked
        return (question_index, question, question_language)

    def _find_language(self, question):
        language = self.language_detector.detect_language_of(question)
        return language.iso_code_639_1

    def _initialise_language_detector(self):
        # Narrow down the dection of the 24 official language of the European Union
        languages = [
            Language.BULGARIAN,
            Language.CROATIAN,
            Language.CZECH,
            Language.DANISH,
            Language.DUTCH,
            Language.ENGLISH,
            Language.ESTONIAN,
            Language.FINNISH,
            Language.FRENCH,
            Language.GERMAN,
            Language.GREEK,
            Language.HUNGARIAN,
            Language.IRISH,
            Language.ITALIAN,
            Language.LATVIAN,
            Language.LITHUANIAN,
            Language.POLISH,
            Language.PORTUGUESE,
            Language.ROMANIAN,
            Language.SLOVAK,
            Language.SLOVENE,
            Language.SPANISH,
            Language.SWEDISH,
        ]
        return LanguageDetectorBuilder.from_languages(*languages).build()
