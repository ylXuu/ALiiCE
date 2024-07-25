import re
import json
import string
from typing import List, Dict


def read_jsonl(file_path: str) -> List[Dict]:
    ''' reading jsonline file '''

    with open(file_path, 'r') as file:
        instances = [json.loads(line.strip()) for line in file if line.strip()]
    
    return instances


def read_json(file_path: str) -> List[Dict]:
    ''' read json file '''

    with open(file_path, 'r') as file:
        instances = json.load(file)
    
    return instances


def normalize_answer(s):
    ''' copied from https://github.com/princeton-nlp/ALCE/blob/main/utils.py '''
    
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent):
    ''' remove citation labels in the sentence '''

    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

