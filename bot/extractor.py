import re

import wikipediaapi
from ctparse import ctparse
from spacy import Language
from word2number import w2n

wiki = wikipediaapi.Wikipedia("en")


def get_numbers(text: str):
    date_removed = re.sub(r"[0-9]+[.-]+[0-9]+[.-]+[0-9]*", "", text)
    numbers = re.findall(r"\d+", date_removed)
    try:
        number_from_text = w2n.word_to_num(date_removed)
        return [*numbers, number_from_text]
    except:
        pass
    return numbers


def get_date(text: str):
    date_result = ctparse(text)
    if date_result and date_result.score > -2000:
        return date_result.resolution.dt
    return None


def get_places(text: str, nlp: Language):
    place_names = []
    previous_token_pos = ""

    for token in nlp(text):
        if token.pos_ == "PROPN":
            if previous_token_pos == "PROPN":
                place_names[-1] += f" {token.lemma_}"
            else:
                place_names.append(token.lemma_)
        previous_token_pos = token.pos_

    results = []
    for place in place_names:
        page_py = wiki.page(place)
        summary = page_py.summary[0:200].lower()

        if "village" in summary or "city" in summary:
            results.append(place)
    return results
