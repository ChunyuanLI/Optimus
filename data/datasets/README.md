# Pre-processing dataset

## Pre-trained dataset: Wikipedia

The dataset can be downloaded here. We split the original wiki text into 299 files, and loop over files in one epoch.

We filter each sentence in wiki based on two constraints: (1) The sentence length is smaller than 64. (2) The tokenized sentence length is smaller than 256 (so that the encoder can take the entire sentence).

To filter the sentence, please change the data folders and run the script:

    sh scripts/scripts_local/run_data_filtering_wiki.sh

The filtered files are saved in "data/datasets/wikipedia_json_64_filtered". There are 298 files in total.


## Fine-tuning datasets

Language Modeling: Penn, Yelp, Yahoo, Snli


(Stylized) Dialog response generation: DailyDialog, Holmes


Label-conditional text generation: Yelp.


Language Understanding: GLUE, Yelp.

