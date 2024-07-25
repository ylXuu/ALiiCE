import re
import json
import copy
import torch
import string
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from myparser import alce_parse_citation, alice_parse_citation
from utils import read_jsonl, read_json, normalize_answer, remove_citations
from typing import Callable
from rouge_score import rouge_scorer, scoring

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
global nli_tokenizer, nli_model
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE_ID = 0


def sentence_spliter(text):
    ''' split text into sentences '''
    sentences = sent_tokenize(text)

    for idx in range(len(sentences)):
        if idx == 0:
            continue

        if re.search(r'\[\d+\]', sentences[idx]) and sentences[idx][0] == '[':
            # the opening citation mark should belong to the previous sentence
            citations = [(match.group(), match.start(), match.end()) for match in re.finditer(r"\[\d+\]", sentences[idx])]
            sentences[idx - 1] = sentences[idx - 1] + citations[0][0]
            sentences[idx] = sentences[idx][0: citations[0][1]] + sentences[idx][citations[0][2]: ]
    
    return sentences


def run_nli(passage, claim, max_length=512):
    ''' run nli model to check whether passage can entail claim
    Copied from https://github.com/princeton-nlp/ALCE/blob/main/eval.py
    '''

    global nli_tokenizer, nli_model

    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = nli_tokenizer(input_text, return_tensors="pt").input_ids.to(nli_model.device)

    if len(input_ids) > 512:
        input_ids = input_ids[:512]

    with torch.no_grad():
        outputs = nli_model.generate(input_ids, max_new_tokens=10)
    
    result = nli_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0

    return inference


def calculate_correctness_asqa(data):
    ''' calculate correctness of asqa using exact match recall '''
    
    def exact_presence(short_answers, context):
        ''' Verify if any of the answers is present in the given context. '''

        n_short_answers = [normalize_answer(sa) for sa in short_answers]
        n_context = normalize_answer(context)

        for ans in n_short_answers:
            if ans in n_context:
                return True

        return False
    
    logging.info('calculating correctness of asqa using exact match recall.')

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0
    
    acc, hit = [], []

    for item in data:
        loc_acc = []
        for qa_pairs in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pairs['short_answers'], item['output']))
        acc.append(np.mean(loc_acc))
        hit.append(int(np.mean(loc_acc) == 1))

    return {
        'str_em': 100 * np.mean(acc), 
        'str_hit': 100 * np.mean(hit),
        'std_em': np.std(acc, ddof=0)
    }



def calculate_correctness_eli5(data):
    ''' calculate correctness of eli5 using rouge-l f1
    Copied from https://github.com/princeton-nlp/ALCE/blob/main/eval.py
    '''
    
    def _rouge_calculation(hypotheses, references1, references2=[], metrics=['rougeLsum']):

        if references2 == []:
            references2 = references1

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for i in range(len(hypotheses)):
            scores1 = scorer.score(references1[i], hypotheses[i])
            scores2 = scorer.score(references2[i], hypotheses[i])
            if scores1['rougeLsum'].fmeasure > scores2['rougeLsum'].fmeasure:
                aggregator.add_scores(scores1)
            else:
                aggregator.add_scores(scores2)

        scores = {m: [] for m in metrics}

        for m in metrics:
            fmeasure = aggregator.aggregate()[m].mid.fmeasure
            scores[m].append(fmeasure)

        for m in scores:
            scores[m] = 100 * sum(scores[m]) / len(scores[m])

        return scores
    
    logging.info('calculating correctness of eli5 using rouge-l f1.')

    hypotheses = {}
    references1 = {}
    references2 = {}

    for idx, item in enumerate(data):
        hypotheses[idx] = item["output"]
        if "annotations" in item and item['annotations'] is not None: # For ASQA
            references1[idx] = item["annotations"][0]["long_answer"]
            references2[idx] = item["annotations"][1]["long_answer"]
        else:
            references1[idx] = item["answer"]
            references2[idx] = item["answer"]

    h, r1, r2 = [], [], []

    for key in references1:
        h.append(hypotheses[key])
        r1.append(references1[key])

        if references2 is not None:
            r2.append(references2[key])

    h = ['\n'.join(sent_tokenize(text.lower())) for text in h]
    r1 = ['\n'.join(sent_tokenize(text.lower())) for text in r1]
    r2 = ['\n'.join(sent_tokenize(text.lower())) for text in r2]
    scores = _rouge_calculation(h, r1, r2)

    return {
        'rouge_l': scores['rougeLsum']
    }


def calculate_mauve(data, model_path='../gpt2-large/'):
    ''' run gpt2-large to calculate fluency
    Copied from https://github.com/princeton-nlp/ALCE/blob/main/eval.py
    '''

    logging.info('calculating mauve.')

    human_data = []
    model_data = []
    for item in data:
        # Remove ending punctuations
        # Remove any new lines
        # Truncate by 100 words
        human_data.append(' '.join((item['question'] + " " + item['answer'].strip()).split()[:100]).rstrip(string.punctuation))
        model_data.append(' '.join((item['question'] + " " + item['output'].strip()).split()[:100]).rstrip(string.punctuation))

    import mauve
    out = mauve.compute_mauve(
        p_text=human_data,
        q_text=model_data,
        device_id=DEVICE_ID,
        max_text_length=512,
        verbose=True,
        batch_size=8,
        featurize_model_name=model_path
    )

    return {
        'mauve': out.mauve * 100
    }


def calculate_citation(data, parser: Callable, psg_num: int=5, doc_key: str='text'):
    ''' calculate fine-grained citation precision and recall '''

    global nli_tokenizer, nli_model
    
    def _format_document(doc, doc_key):

        if 'title' in doc:
            if doc_key in doc:
                return 'Title: %s\n%s' % (doc['title'], doc[doc_key])
            if 'text' in doc:
                return 'Title: %s\n%s' % (doc['title'], doc['text'])

        return doc
    
    ais_scores = []
    ais_scores_prec = []

    area_total = 0
    area_mcite = 0
    area_mcite_support = 0
    area_mcite_overcite = 0

    num_no_aliice = 0
    
    for item in tqdm(data, desc='evaluating citation recall and precision.'):
        sentences = sentence_spliter(item['output'])

        if len(sentences) == 0:
            continue

        entail = 0
        num_areas = 0
        entail_prec = 0
        total_citations = 0

        for sentence in sentences:
            parsed_citation = parser(sentence)

            cite_text_pairs = []

            if type(parsed_citation) is tuple:
                # alce 
                cite_text_pairs.append(parsed_citation)
                num_no_aliice += 1

            elif type(parsed_citation) is dict:
                # aliice
                for k, v in parsed_citation.items():
                    cite_text_pairs.append((k, v))
            
            # the difference is that a sentence might have more than one citation positions
            for cite_text_pair in cite_text_pairs:
                joint_entail = -1

                text = cite_text_pair[0]
                cite = cite_text_pair[1]
                
                if len(cite) == 0:
                    # no citation
                    joint_entail = 0
                elif any([cite_id > psg_num for cite_id in cite]):
                    # citation index out of range
                    joint_entail = 0
                else:
                    num_areas += 1
                    total_citations += len(cite)
                    joint_passage = '\n'.join([_format_document(item['docs'][pid - 1], doc_key) for pid in cite])
                
                # calculate recall
                if joint_entail == -1:
                    joint_entail = run_nli(joint_passage, text)
                
                entail += joint_entail

                if len(cite) > 1:
                    area_mcite += 1
                
                # calculate precision if recallable
                if joint_entail and len(cite) > 1:
                    area_mcite_support += 1

                    for pid in cite:
                        passage = _format_document(item['docs'][pid - 1], doc_key)
                        nli_result = run_nli(passage, text)

                        if not nli_result:
                            subset_exclude = copy.deepcopy(cite)
                            subset_exclude.remove(pid)
                            passage = '\n'.join([_format_document(item['docs'][pid - 1], doc_key) for pid in subset_exclude])
                            nli_result = run_nli(passage, text)

                            if nli_result:
                                # docs[pid] is not neccesary
                                area_mcite_overcite += 1
                            else:
                                entail_prec += 1
                        else:
                            entail_prec += 1
                else:
                    entail_prec += joint_entail
        
        if num_areas == 0:
            continue
        
        area_total += num_areas
        ais_scores.append(entail / num_areas)
        ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0)

    
    if area_mcite > 0 and area_mcite_support > 0:
        logging.info("Among all citation area, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
            100 * area_mcite / area_total, 
            100 * area_mcite_support / area_mcite, 
            100 * area_mcite_overcite / area_mcite_support
        ))
    
    return {
        'citation_recall': 100 * np.mean(ais_scores),
        'citation_precision': 100 * np.mean(ais_scores_prec),
        'recall_std': np.std(ais_scores, ddof=0),
        'precision_std': np.std(ais_scores_prec, ddof=0),
        'times_not_used_aliice': num_no_aliice
    }


def calculate_cv(data):
    ''' calculate coefficient of variation of citation position in a sentence '''

    logging.info('calculating coefficient of variation of citation position.')
    
    cvs = []

    for item in tqdm(data, desc='Evaluating coefficient of variation of citation position.'):
        sentences = sentence_spliter(item['output'])
        for sentence in sentences:

            if re.search(r'\[\d+\]', sentence) is None:
                # this sentence has no citation labels, so we skip it
                continue

            cite_pos = [match.start() / len(sentence) for match in re.finditer(r"\[\d+\]", sentence)]
            cvs.extend(cite_pos)

    return {
        'cv': np.std(cvs, ddof=0) / (sum(cvs) / len(cvs))
    }


def calculate_length(data):
    ''' calculate average length of response '''

    logging.info('calculating average length of response.')

    list_len = []
    for item in data:
        list_len.append(len(item['output'].split()))
    
    return {
        'length_std': np.std(list_len, ddof=0),
        'avg_length': sum(list_len) / len(list_len)
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='The output file to be evaluated.')
    parser.add_argument('--benchmark', choices=['alce', 'aliice'], default='aliice', help='The selected benchmark.')
    parser.add_argument('--dataset', choices=['asqa', 'eli5'], help='The QA dataset used to generate output.')
    parser.add_argument('--output_path', required=True, help='The output file directory to save the evaluation result.')
    parser.add_argument('--correctness', action='store_true', help='Whether needs to calculate correctness.')
    parser.add_argument('--mauve', action='store_true', help='Whether needs to calculate mauve.')
    parser.add_argument('--length', action='store_true', help='Whether needs to calculate average length of response.')
    parser.add_argument('--cv', action='store_true', help='Whether needs to calculate coefficient variation of citation position.')
    parser.add_argument('--citation', action='store_true', help='Whether needs to calculate citation quality.')
    parser.add_argument('--psg_num', type=int, default=5, help='The number of passages used in generation.')
    parser.add_argument('--use_sum', action='store_true', help='Use passage\'s summary.')
    parser.add_argument('--use_snippet', action='store_true', help='Use passages\' snippet.')
    parser.add_argument('--fg_only', action='store_true', help='Whether only consider the fine-grained response')
    parser.add_argument('--nli_model_path', help='The NLI model path.')
    parser.add_argument('--mauve_model_path', help='The path of the model used to calculate mauve.')

    args = parser.parse_args()

    with open(args.data_path, 'r') as file:
        data = json.load(file)
    
    # remove all citation labels for all non-AutoAIS evaluation
    normalize_data = copy.deepcopy(data)
    for i in range(len(normalize_data)):
        normalize_data[i]['output'] = remove_citations(normalize_data[i]['output'])
    
    result = {}
    if args.length:
        result.update(calculate_length(normalize_data))
    if args.cv:
        result.update(calculate_cv(data))
    if args.mauve:
        result.update(calculate_mauve(normalize_data, model_path=args.mauve_model_path))
    if args.correctness:
        if args.dataset == 'asqa':
            result.update(calculate_correctness_asqa(normalize_data))
        else:
            result.update(calculate_correctness_eli5(normalize_data))
    if args.citation:
        global nli_tokenizer, nli_model
        nli_model = AutoModelForSeq2SeqLM.from_pretrained(args.nli_model_path, torch_dtype=torch.bfloat16, device_map='auto')
        nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model_path)

        if args.benchmark == 'alce':
            result.update(calculate_citation(data, alce_parse_citation))
        else:
            result.update(calculate_citation(data, alice_parse_citation))

    with open(args.output_path, 'w') as file:
        json.dump(result, file, indent=4)
    

if __name__ == '__main__':
    main()

