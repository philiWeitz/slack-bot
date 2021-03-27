import pathlib
import pickle
import random
import re
from itertools import groupby

import numpy as np
import spacy
from keras.models import load_model
from pydash import chain, is_empty, reject

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
    "nextState": "",
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

    return {
        "answer": " ".join(answers),
        "intent": intents_json[largest_index[-1]]["tag"],
        "nextState": intents_json[largest_index[-1]]["nextState"],
    }


# print(get_response("Hello!"))
# print(get_response("How are you?"))
# print(get_response("Hello! How are you?"))
# print(get_response("I'm ok"))
# print(get_response("See you later alligator"))
# print(get_response("Hello! See you later alligator"))
#
#
# print(get_response("Hello! How are you?"))
# print(get_response("I'm ooooookkkkk"))
# print(get_response("So what can I do?"))
# print(get_response("See you later alligator"))
# print(get_response("It's raining"))
