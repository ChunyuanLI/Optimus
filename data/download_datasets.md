# Download/Pre-process Datasets

## Wikipedia

Option1: Download processed files (11.78G) below, and unzip it (298 files)

https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/wikipedia_json_64_filtered.zip

Option2: Download raw file (11.79G):

https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/wikipedia.segmented.nltk.txt

Our pre-processing protocal: We split the original wiki text into 298 files, and loop over files in one epoch. We filter each sentence in wiki based on two constraints: (1) The sentence length is smaller than 64. (2) The tokenized sentence length is smaller than 256 (so that the encoder can take the entire sentence). To filter the sentence, please change the data folders and run the script:

    sh scripts/scripts_local/run_data_filtering_wiki.sh

The filtered files are saved in "data/datasets/wikipedia_json_64_filtered".


## Fine-tuning datasets

Language Modeling: Penn, Yelp, Yahoo, Snli

https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/penn_data.zip
https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/yelp_data.zip
https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/yahoo_data.zip
https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/snli_data.zip

A tiny dataset is also provided for the purpose of debugging
https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/debug_data.zip

(Stylized) Dialog response generation: DailyDialog, Holmes


Label-conditional text generation: Yelp.


Language Understanding: GLUE, Yelp.

