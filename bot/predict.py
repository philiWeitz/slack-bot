import pathlib
import pickle
import random
import re
from itertools import groupby

import numpy as np
import spacy
from keras.models import load_model
from pydash import chain, is_empty, reject

from bot.extractor import get_date, get_numbers, get_places
from bot.train import lemmatize_sentence, sentence_to_feature_vector

NLP_FOLDER = pathlib.Path(__file__).parent.absolute()
MODEL_FOLDER = pathlib.Path.joinpath(NLP_FOLDER, "model")

PUNCTUATION_REGEX = "[.?!]{1}"

nlp = spacy.load("en_core_web_sm")

model = load_model(f"{MODEL_FOLDER}/chatbot.h5")
bag_of_words = pickle.load(open(f"{MODEL_FOLDER}/words.pkl", "rb"))
intents_json = pickle.load(open(f"{MODEL_FOLDER}/intents.pkl", "rb"))

MINIMUM_THRESHOLD = 0.7

NO_INTENT_DETECTED_ANSWER = {
    "answer": "Sorry, I didn't understand you.",
    "intent": "no-intent",
    "nextStates": [],
}


def clean_sentence(sentence: str):
    return "".join(c for c, _ in groupby(sentence.strip()))


def get_bot_response(text):
    sentences = (
        chain(re.split(PUNCTUATION_REGEX, text))
        .reject(is_empty)
        .map(clean_sentence)
        .value()
    )

    if len(sentences) <= 0:
        return NO_INTENT_DETECTED_ANSWER

    lemmatized_sentences = reject(
        [lemmatize_sentence(nlp(sentence)) for sentence in sentences],
        is_empty,
    )

    X = [
        sentence_to_feature_vector(sentence, bag_of_words)
        for sentence in lemmatized_sentences
    ]

    predictions = model.predict(X)
    largest_index = np.argmax(predictions, axis=1)

    largest_values = np.take_along_axis(
        predictions, np.expand_dims(largest_index, axis=-1), axis=-1
    )

    # last user intent was not detected correctly
    if largest_values[-1][0] < MINIMUM_THRESHOLD:
        return NO_INTENT_DETECTED_ANSWER

    answers = [
        random.choice(intents_json[idx]["responses"]) for idx in largest_index
    ]

    for idx in largest_index:
        if "date" in intents_json[idx]["extract"]:
            extracted_date = get_date(sentences[-1])
            answers[-1] = answers[-1].replace("<date>", str(extracted_date))
        if "place" in intents_json[idx]["extract"]:
            extracted_places = get_places(sentences[-1], nlp)
            answers[-1] = answers[-1].replace(
                "<place>", ",".join(extracted_places)
            )
        if "number" in intents_json[idx]["extract"]:
            extracted_numbers = get_numbers(sentences[-1])
            answers[-1] = answers[-1].replace(
                "<number>", ",".join(extracted_numbers)
            )

    return {
        "answer": " ".join(answers),
        "intent": intents_json[largest_index[-1]]["tag"],
        "nextStates": intents_json[largest_index[-1]]["nextStates"],
    }


print(get_bot_response("I take 10"))
