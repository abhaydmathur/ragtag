""" Preprocessing user input """
from lingua import Language, LanguageDetectorBuilder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import sys
import logging
import pandas as pd


class Configuration:
    def __init__(self, loglevel, csv_file, input_question, output_file, dataset_documents, rag_model_path):
        self.logger = self.setup_logging(loglevel)
        self.available_translators_to_eng = {}
        self.dataset_documents_path = dataset_documents
        self.rag_model = rag_model_path
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
        question_index = next(iter(self.questions), None)
        print(question_index)
        if question_index is None:
            return None
        question = self.questions[question_index]
        question_language = self._find_language(question)
        # return the id, question, and the language in which the question was asked
        del self.questions[question_index]
        return (question_index, question, question_language)

    def load_translation_model(self):
        model_path = "/gpfsdswork/dataset/HuggingFace_Models/facebook/nllb-200-3.3B"
        self.translater_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.translated_tokenizer = AutoTokenizer.from_pretrained(model_path)

    def load_translation_pipeline_to_eng(self, language):
        print(self.available_translators_to_eng)
        if language in self.available_translators_to_eng:
            return self.available_translators_to_eng[language]
        else:
            translator = pipeline(
                "translation",
                model=self.translater_model,
                tokenizer=self.translated_tokenizer,
                src_lang=language,
                tgt_lang="eng_Latn",
            )
            self.available_translators_to_eng[language] = translator
            return translator

    def translation_to_eng(self, question, src_language):
        self.logger.info(
            f"Translating question, {question}, from {src_language} to 'eng_Latn'."
        )
        translator = pipeline(
            "translation",
            model=self.translater_model,
            tokenizer=self.translated_tokenizer,
            src_lang=src_language,
            tgt_lang="eng_Latn",
        )
        # translator = self.load_translation_pipeline_to_eng(src_language)
        output = translator(question, max_length=400)
        output = output[0]["translation_text"]
        return output

    def _find_language(self, question):
        language = self.language_detector.detect_language_of(question)
        return language.iso_code_639_3

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
