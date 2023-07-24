  # spaCy implementation of 5W1H Extraction using the Giveme5W1H package by Hamborg et al. 2019

  ## Requirements
  ```
  python 3.8
  ```
  
  ## Installation
  ```
  pip install -r requirements.txt
  pip install git+https://github.com/hitt08/neuralcoref.git

  python -m spacy download en_core_web_sm
  python -c "import benepar; benepar.download('benepar_en3')"
  python -c "import nltk; nltk.download('wordnet','./data/nltk_data'); nltk.download('stopwords','./data/nltk_data'); nltk.download('omw-1.4','./data/nltk_data')"
  ```

  ## Getting Started

  5W1H Extraction
  ```
  python extract_5w1h.py [-p <N_PROCESSES>] [-m <NLP_MODEL>] [--use_cpu] [-f] [--no_threading] [--skip_where] [--errors]
                         [--hide_errors]
  
  Arguments:
    -p <N_PROCESSES>  Number of processes to use
    -m <NLP_MODEL>    spaCy model to use, or coreNLP
    --use_cpu         Use CPU
    -f                Force reprocessing
    --no_threading    Disable multithreaded extraction
    --skip_where      Skip where extraction
    --errors          exit on errors
    --hide_errors     hide errors

  e.g.
  python extract_5w1h.py --use_cpu -p 1
  ```

  Vectorisation
  ```
  python vectorise_5w1h.py [--lsa_dim <LSA_DIM>] [--skip_tfidf] [--skip_minilm] [--skip_roberta]
                         [--sbert_batch <SBERT_BATCH>] [--use_cpu]

  Arguments:
  --lsa_dim <LSA_DIM>         LSA Dimension
  --skip_tfidf                Skip TFIDF Vectorise
  --skip_minilm               Skip MiniLM Vectorise
  --skip_roberta              Skip Roberta Vectorise
  --sbert_batch <SBERT_BATCH> SBERT Batch Size
  --use_cpu                   Use CPU

  e.g.
  python vectorise_5w1h.py --use_cpu
  ```

  <br><br>
  ## Giveme5W1H:
  https://github.com/fhamborg/Giveme5W1H
  ```
  @InProceedings{Hamborg2019b,
    author    = {Hamborg, Felix and Breitinger, Corinna and Gipp, Bela},
    title     = {Giveme5W1H: A Universal System for Extracting Main Events from News Articles},
    booktitle = {Proceedings of the 13th ACM Conference on Recommender Systems, 7th International Workshop on News Recommendation and Analytics (INRA 2019)},
    year      = {2019},
    month     = {Sept.},
    location  = {Copenhagen, Denmark}
  }
  ```
  
