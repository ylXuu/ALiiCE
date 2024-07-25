# Data

We directly use the dataset from [ALCE(Hugging Face)](https://huggingface.co/datasets/princeton-nlp/ALCE-data). This dataset includes three QA datasets, which are ASQA, ELI5 and QAMPARI. We only ASQA and ELI5 for these two are typical long-form QA datasets. Besides, you can also use other long-form QA datasets and format them as shown below.

The fields we used in our experiment of the two datasets are as follows:

ASQA
- question
- docs
    - id
    - title
    - text
    - score
    - summary
    - extraction: the snippet version of document text
- answer: the reference answer
- output: the generated answer

ELI5
- question
- answer: the reference answer
- docs
    - title
    - text
    - summary
    - extraction: the snippet version of document text
- output: the generated answer


