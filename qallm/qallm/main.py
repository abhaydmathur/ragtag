import argparse
import sys
import os
from .configuration import Configuration

from .retrieval_mod import loadRAG, batch_inf

def get_ressource_url(path_md):
    with open(path_md) as f:
        first_line = f.readline()
        return first_line.split(" ")[3]

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
        metavar="path",
        type=str,
        dest="output_file",
        nargs=1,
        help="relative path to the output directory in which the output shall be written to. If not defined, the prediction results will be displayed on the standard output.",
    )
    parser.add_argument(
        "-d",
        "--datatet-documents",
        metavar="path",
        type=str,
        dest="dataset_documents",
        nargs=1,
        help="relative path to the directory of the document dataset.",
    )
    parser.add_argument(
        "-r",
        "--rag-model-path",
        metavar="path",
        type=str,
        dest="rag_model_path",
        nargs=1,
        help="relative path to the Rag Model.",
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
        args.loglevel, args.csv_file, args.input_question, args.output_file,args.dataset_documents, args.rag_model_path
    )
    current_entry = config.get_next_element()
    questions = {}
    # config.load_translation_model()
    # while current_entry:
    #     question_index, question, question_language = current_entry
    #     if question_language.name != "ENG":
    #         question_english = config.translation_to_eng(
    #             question, question_language.name.lower() + "_Latn"
    #         )
    #         questions[question_index] = {
    #             "question": question,
    #             "question_english": question_english,
    #             "source_language": question_language.name,
    #         }
    #     else:
    #         questions[question_index] = {
    #             "question": question,
    #             "question_english": question,
    #             "source_language": "EN",
    #         }
    #     current_entry = config.get_next_element()

    # config.logger.info(f"Loaded and translated all {questions}")

    questions = {
        3: {
            "question": "How many Italian government data requests did LinkedIn receive in 2022? Please provide the URL of the source.",
            "question_english": "How many Italian government data requests did LinkedIn receive in 2022? Please provide the URL of the source.",
            "source_language": "EN",
        },
        0: {
            "question": 'Écris-moi une requête curl. Cette requête doit me permettre d\'obtenir, auprès de Facebook, les publicités politiques contenant le mot "europe" et qui ont atteint la France et la Belgique. La réponse ne doit contenir que le code de la requête curl.',
            "question_english": 'Write me a curl request. This request should allow me to obtain from Facebook the political advertisements containing the word "Europe" and which have reached France and Belgium. The response should contain only the code of the curl request.',
            "source_language": "FRA",
        },
        1: {
            "question": "Comment savoir si une campagne publicitaire sur Facebook a dévié de sa cible prévue ? La réponse doit contenir un lien vers la documentation des API de Facebook au sujet des publicités. La réponse doit nommer les champs les plus pertinents disponibles via cette API.",
            "question_english": "How to tell if an ad campaign on Facebook has deviated from its intended target? The response should contain a link to the Facebook API documentation about ads. The response should name the most relevant fields available through that API.",
            "source_language": "FRA",
        },
        2: {
            "question": "Où puis-je trouver le centre de transparence de Linkedin ? Veuillez indiquer une seule URL.",
            "question_english": "Where can I find the Linkedin Transparency Centre? Please provide a single URL.",
            "source_language": "FRA",
        },
        4: {
            "question": "How many languages spoken in the EU are spoken by the Pinterest team ? Can you specify the link where this information is mentioned ?",
            "question_english": "How many languages spoken in the EU are spoken by the Pinterest team ? Can you specify the link where this information is mentioned ?",
            "source_language": "EN",
        },
        5: {
            "question": "I am a coding expert. How could I get more info about a Facebook post whose url slug ends with 123456789_123456789? I have a crowdtangle API token (TOKEN). Your output is a bash code snippet.",
            "question_english": "I am a coding expert. How could I get more info about a Facebook post whose url slug ends with 123456789_123456789? I have a crowdtangle API token (TOKEN). Your output is a bash code snippet.",
            "source_language": "EN",
        },
        6: {
            "question": "I have access to a set of tweets URLs that I consider to be hateful. How can I use Twitter's API to monitor the average duration between the tweet's creation and its moderation?",
            "question_english": "I have access to a set of tweets URLs that I consider to be hateful. How can I use Twitter's API to monitor the average duration between the tweet's creation and its moderation?",
            "source_language": "EN",
        },
        7: {
            "question": "How to get the number of likes of a TikTok post. I'd like an example query and an explanation of the parameters used.",
            "question_english": "How to get the number of likes of a TikTok post. I'd like an example query and an explanation of the parameters used.",
            "source_language": "EN",
        },
        8: {
            "question": "Comment je peux contacter le représentant légal du DSA de Snapchat ?",
            "question_english": "How do I contact the legal representative of the Snapchat DSA?",
            "source_language": "FRA",
        },
        9: {
            "question": "How is content moderation carried out on X?",
            "question_english": "How is content moderation carried out on X?",
            "source_language": "EN",
        },
    }

    answers = {}


    ret, model, tokenizer, doc_data = loadRAG(args.dataset_path, args.rag_model_path)

    responses, doc_ids = batch_inf([questions[id] for id in questions.keys()], ret, model, tokenizer)

    doc_files = [doc_data[id]["Title"] for id in doc_ids]
    # Translate back to origin language the answer.

    urls = [get_ressource_url(fname) for fname in doc_files]

    # Path 
    print(config.output_file)
    answers_path = os.path.join(config.output_file[0], "answers") 
    os.makedirs(answers_path, exist_ok=True)
    prompts_path = os.path.join(config.output_file[0], "prompts") 
    os.makedirs(prompts_path, exist_ok=True)
    sources_path = os.path.join(config.output_file[0], "sources")
    os.makedirs(sources_path, exist_ok=True)

    for key in questions.keys():
        fp = open(os.path.join(answers_path,f'{key}.txt'), 'w')
        fp.write("You have been rickrolled .)")
        fp.close()
        
        fp = open(os.path.join(prompts_path,f'{key}.txt'), 'w')
        fp.write(questions[key]["question"])
        fp.close()

        fp = open(os.path.join(sources_path,f'{key}.txt'), 'w')
        fp.write("https://peren.gouv.fr")
        fp.close()

def main():
    cli(sys.argv[1:])
