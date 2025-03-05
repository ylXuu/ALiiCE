# ALiiCE
This is the repository of **ALiiCE** (**A**utomatic **L**LM's Pos**i**tional F**i**ne-grained **C**itation **E**valuation), from the paper: [ALiiCE: Evaluating Positional Fine-grained Citation Generation.](https://arxiv.org/abs/2406.13375)

## News

- Our paper was accepted by NAACL 2025 Main Conference!

## Introduction
ALiiCE focuses on the task of positional fine-grained citation text generation, which citation marks can appear anywhere within sentences. Our framework first parses the sentence claim into atomic claims via dependency analysis. We implement three novel metrics, including positional fine-grained citation recall and precision, and coefficient of variation of citation positions. Our code can be a effective method to evaluate performance for the task of positional fine-grained citation text generation.



The function of each main file or folder is as follows:
- data/: ASQA and ELI5 datasets
- prompts/: prompts used in LLM generation
- eval.py: evaluate the output of LLM generation
- generate.py: generate output file of LLM generation
- myparser.py: the implementation of ALiiCE parsing algorithm


## Installation
All requirements are displayed at [requirements.txt](requirements.txt) and you can setup the environment by running the following command of conda:
```bash
conda env create -f requirements.yml
```
the main packages we used are: `python>=3.10`, `torch==2.2.2+cu118`, `transformers`, `spacy`, `pandas`, `nltk`, `rouge_score`, `openai`, `graphviz`.

Then, run the following command to download the English models of `spacy`:
```bash
python -m spacy download en_core_web_sm
```

## Dataset
You can refer to [./data/README.md](data/README.md) to obtain the datasets required for generation and evaluation.


## Generation
We provide two types of LLM generation implementations, which are OpenAI's LLMs and llama-3 series.

```bash
python generate.py --data_path ./data/data.json \
    --prompt_config ./prompts/generation.json \
    --output_path ./outputs/output.json \
    --psg_num 5 \
    --model_name llama-3-8b
```
the explanation of every parameter is as follows:
- --data_path: the path of dataset file
- --prompt_config: the prompt file used for generation
- --output_path: the path of file to store the output
- --model_name: the model name used for generation. You can use `gpt-xxx` or `llama-3-xxx`
- --psg_num: the number of retrieved passages used in the generation of response
- --use_sum: use this when the passage's format is summary
- --use_snippet: use this when the passage's format is snippet



## Evaluation
After the generation, you can evaluate the performance of your model's output by our code.For example, if you want to evaluate the output on ELI5 using ALiiCE, you can run the following command:
```bash
python eval.py --data_path ./outputs/output.json \
    --benchmark aliice \
    --dataset eli5 \
    --output_path ./results/result.json \
    --citation \
    --correctness \
    --mauve \
    --length \
    --cv \
    --psg_num 5 \
```
the explanation of every parameter is as follows:
- --data_path: the path of output file to be evaluated
- --benchmark: the evaluation method, which has two choices: ALCE and ALiiCE
- --dataset: the used dataset, which has two choices: ASQA and ELI5
- --output_path: the json file path to save the evaluation result
- --citation: evaluate citation quality, including citation recall and citation precision
- --correctness: evaluate correctness, which employ exact string match for ASQA and employ ROUGE-L for ELI5
- --mauve: evaluate fluency using [MAUVE](https://arxiv.org/abs/2102.01454)
- --length: evaluate average length of response
- --cv: evaluate coefficient variation of citation positions, which indicates the dispersion of citation placements within a sentence. Please refer to our paper for a detailed description
- --psg_num: the number of retrieved passages used in the generation of response
- --use_sum: use this when the passage's format is summary
- --use_snippet: use this when the passage's format is snippet



## Acknowledgement
We would thank for the [ALCE](https://github.com/princeton-nlp/ALCE) repository for providing guidance or inspiration.


## Citation

If ALiiCE is useful to you, please cite the following paper in your work:

```bibtex
@misc{xu2024aliiceevaluatingpositionalfinegrained,
      title={ALiiCE: Evaluating Positional Fine-grained Citation Generation}, 
      author={Yilong Xu and Jinhua Gao and Xiaoming Yu and Baolong Bi and Huawei Shen and Xueqi Cheng},
      year={2024},
      eprint={2406.13375},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.13375}, 
}
```

