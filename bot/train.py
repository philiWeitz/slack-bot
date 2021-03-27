import json
import pathlib
import pickle

import numpy as np
import spacy
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from pydash import chain, flatten, is_empty


NLP_FOLDER = pathlib.Path(__file__).parent.absolute()
MODEL_FOLDER = pathlib.Path.joinpath(NLP_FOLDER, "model")

# spacy's stop words seem to be a bit too restrictive
STOP_WORDS = ["if", "a", "is"]

intents_json = json.loads(
    open(pathlib.Path.joinpath(NLP_FOLDER, "intents.json")).read()
)


def lemmatize_sentence(sentence):
    punctuation_removed = [word for word in sentence if not word.is_punct]
    lemmatized = [word.lemma_ for word in punctuation_removed]
    return [word for word in lemmatized if not word in STOP_WORDS]


def sentence_to_feature_vector(sentence, bag_of_words):
    result_vector = list(np.zeros(len(bag_of_words)))
    for word in sentence:
        if word in bag_of_words:
            result_vector[bag_of_words.index(word)] = 1
    return result_vector


def train_model():
    nlp = spacy.load("en_core_web_sm")

    sentences = flatten([item["patterns"] for item in intents_json])
    sentences = [nlp(sentence) for sentence in sentences]
    sentences = [lemmatize_sentence(sentence) for sentence in sentences]

    bag_of_words = chain(sentences).flatten().uniq().sort().value()

    intents = [item["patterns"] for item in intents_json]

    X = [
        sentence_to_feature_vector(sentence, bag_of_words)
        for sentence in sentences
    ]

    y = []
    for idx, patterns in enumerate(intents):
        for pattern in patterns:
            entry = list(np.zeros(len(intents)))
            entry[idx] = 1
            y.append(entry)

    indexes = [i for i, x in enumerate(sentences) if is_empty(x)]
    for index in indexes:
        del X[index]
        del y[index]

    model = Sequential()
    model.add(Dense(64, input_shape=(len(X[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(y[0]), activation="softmax"))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
    )

    # fitting and saving the model
    hist = model.fit(
        np.array(X), np.array(y), epochs=200, batch_size=5, verbose=1
    )

    pickle.dump(bag_of_words, open(f"{MODEL_FOLDER}/words.pkl", "wb"))
    pickle.dump(intents_json, open(f"{MODEL_FOLDER}/intents.pkl", "wb"))
    model.save(f"{MODEL_FOLDER}/chatbot.h5", hist)


if __name__ == "__main__":
    train_model()
