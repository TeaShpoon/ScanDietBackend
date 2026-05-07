from nltk.tokenize.util import align_tokens
from collections import defaultdict


def reverse_ner(ner_output):
    entities = defaultdict(list)
    for token, label in ner_output:
        if label == "O":
            continue
        etype = label.split("-")[-1].lower()
        entities[etype].append(token)
    return dict(entities)


def get_spans(prediction, sentence):
    tokens, labels = zip(*prediction)
    return list(zip(align_tokens(tokens, sentence), labels))
