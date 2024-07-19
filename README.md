# qachatbot

A simple QA chatbot

## Setup

We recommed using miniconda, although every other python virtual env managers will work fine.

The command below will install a conda environment with the name `qachatbot`. You can change that name in `environment.yml` before running it.

For conda user:

```shell
conda env create -f environment.yml 
conda activate qachatbot
```

For other user:

```shell
pip install -r requirements.txt
```

Install nltk packages:

```shell
import nltk
nltk.download()
```

To run the chat interface, run:

```shell
chainlit run app.py -w
```