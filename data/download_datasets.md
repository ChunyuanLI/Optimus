# Download/Pre-process Datasets

## Wikipedia

Option |  Files    | Size | Data |
| -------- | ------- |  -------- | ------- |
|1 | Processed Files in Zip  | 11.78G| [Download](https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/wikipedia_json_64_filtered.zip?sp=r&st=2023-08-28T00:40:43Z&se=3023-08-28T08:40:43Z&sv=2022-11-02&sr=c&sig=kUkSFqeHFfTeqxxpvqVdICCJupwODFwJprCAW2o4irE%3D)    |
|2 | Raw Text  | 11.79G| [Download](https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/wikipedia.segmented.nltk.txt?sp=r&st=2023-08-28T00:40:43Z&se=3023-08-28T08:40:43Z&sv=2022-11-02&sr=c&sig=kUkSFqeHFfTeqxxpvqVdICCJupwODFwJprCAW2o4irE%3D)      |


Our pre-processing protocal: We split the original wiki text into 298 files, and loop over files in one epoch. We filter each sentence in wiki based on two constraints: (1) The sentence length is smaller than 64. (2) The tokenized sentence length is smaller than 256 (so that the encoder can take the entire sentence). To filter the sentence, please change the data folders and run the script:

    sh scripts/scripts_local/run_data_filtering_wiki.sh

The filtered files are saved in "data/datasets/wikipedia_json_64_filtered".


## Fine-tuning datasets

Language Modeling: Penn, Yelp, Yahoo, Snli. A tiny dataset is also provided for the purpose of debugging

Dataset |  Files    | 
| -------- | ------- |
| Penn | [Zip](https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/penn_data.zip?sp=r&st=2023-08-28T00:40:43Z&se=3023-08-28T08:40:43Z&sv=2022-11-02&sr=c&sig=kUkSFqeHFfTeqxxpvqVdICCJupwODFwJprCAW2o4irE%3D)|
| Yelp | [Zip](https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/yelp_data.zip?sp=r&st=2023-08-28T00:40:43Z&se=3023-08-28T08:40:43Z&sv=2022-11-02&sr=c&sig=kUkSFqeHFfTeqxxpvqVdICCJupwODFwJprCAW2o4irE%3D)|
| Yahoo | [Zip](https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/yahoo_data.zip?sp=r&st=2023-08-28T00:40:43Z&se=3023-08-28T08:40:43Z&sv=2022-11-02&sr=c&sig=kUkSFqeHFfTeqxxpvqVdICCJupwODFwJprCAW2o4irE%3D) |
| Snli | [Zip](https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/snli_data.zip?sp=r&st=2023-08-28T00:40:43Z&se=3023-08-28T08:40:43Z&sv=2022-11-02&sr=c&sig=kUkSFqeHFfTeqxxpvqVdICCJupwODFwJprCAW2o4irE%3D) |
| Debug | [Zip](https://chunylcus.blob.core.windows.net/machines/msrdl/optimus/data/datasets/debug_data.zip?sp=r&st=2023-08-28T00:40:43Z&se=3023-08-28T08:40:43Z&sv=2022-11-02&sr=c&sig=kUkSFqeHFfTeqxxpvqVdICCJupwODFwJprCAW2o4irE%3D) |


