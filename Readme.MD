# Supplementary Material for InfiniSearch:First Stage Retrieval with Byte Suffix Arrays at Pre-training Scale

## Run the pipeline
Install the requirements first. You need to create an infinigram_index from the MSMarco-Dataset.
For references, see:
https://infini-gram.readthedocs.io/en/latest/indexing.html
Here is the dataset:
https://trec-rag.github.io/annoucements/2025-rag25-corpus/
We used the segmented version of the dataset.

You can use `run_experiements.run_from_config(path)` to run a pipeline over the msmarco-index.
with `beir_setup.py`, you can create infinigram-idices from beir datasets. link to their qrels and topics and run the pipeline.

`qual_analysis.py` provides you with an opportunity to query the infinigram api and create extensions from an API in one go.

Put your tokens in a `token_file.ini`

`./qualitative_analysis`, contains the full rankings of all retrieved documents from the qualitative analysis.