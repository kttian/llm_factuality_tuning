# Fine-tuning Language Models for Factuality (FactTune)

## What is this repo?
This repo includes a reference implementation of FactTune, a method to fine-tune langauge models for improved factuality, from our paper [Fine-tuning Language Models for Factuality](https://arxiv.org/abs/2311.08401) at ICLR 2024. 

This repo will walk you through the three main steps: truthfulness scoring, training with [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), and evaluation. Additionally, we provide the code we used to generate our datasets, so that you can use it for your own tasks as well.

## Environment Set Up
First, set the environment variables needed in this codebase by modifying `init_environ.sh` and running:
```
source init_environ.sh 
```

Next, set up the python environments. This repo supports two types of truthfulness scoring: factscoring and confidence scoring. We use two separate python environments for factscoring and confidence scoring (fsenv and csenv, respectively) since they use different versions of pytorch. Python 3.8+ is recommended for both.

For fsenv:
```
python3.8 -m venv fsenv
source fsenv/bin/activate
pip install -r requirements_fsenv.txt
pip install sentencepiece
pip install --upgrade pytz
pip install spacy
python -m spacy download en_core_web_sm
```
Note that this step installs our custom fork of FactScore. You'll need to download some data for it (command below). However, please see the README here https://github.com/kttian/FActScore to learn more about FactScore and context on setting it up (especially if you encounter issues).
```
python -m factscore.download_data --llama_7B_HF_path "huggyllama/llama-7b" --data_dir $FACTSCORE_CACHE_BASE --model_dir $FACTSCORE_CACHE_BASE
```

For csenv:
```
python3.8 -m venv csenv
source csenv/bin/activate
pip install -r requirements_csenv.txt
```

Additional Files:
- OpenAI API Key: Both factscore and confidence scoring atomic use the openai API, so copy your openai API key into the `openai_key.txt` file. Be careful not to git commit or push your key! 
- HuggingFace: Llama2 requires a huggingface authentication token. Put it in `use_hf_auth_token.txt`.
- Llama2-Chat System Prompt: Our default system prompt is here `system_prompt_concise.txt`. 


## Truthfulness Scoring
Given a dataset of prompts and multiple responses per prompt, this step computes a truthfulness score for each response.

Inside `scoring/`, we provide scripts for 3 types of scores:
- factscoring, based on [Min et al.](https://arxiv.org/abs/2305.14251)
- confidence scoring (atomic)
- confidence scoring (entity)

### Compute factscore
See `scoring/factscoring/compute_factscore_bio.sh` as an example for how to run factscoring (FS). 
The outputs are saved as one file per topic under `$DATA_DIR/factscore_outputs`.

### Compute confidence scores
In order to compute the model confidence score in a response, we extract either atomic questions (AQ) or named entities (NE) and resample them from the model. 

Preparing samples for atomic confidence scoring:

- `atomic_fact_to_q.py`: Convert atomic facts from factscore outputs (from previous section) into atomic questions
- `atomic_sample_answers.py`: Sample answers to atomic questions

Preparing samples for entity scoring (optional):

- `entity_sample.py`: Extract entities and resample

Finally, we load the samples computed above and compute confidence scores from them:

- `score_metrics.py`: Compute the confidence scoring metrics and aggregate all types of scores

The output will be one file of FS, AQ, and NE (optional) scores ready to be passed into DPO training. The file is saved to `$DATA_DIR/rewards/rewards_{split}_{len}.json`.

Additionally, it saves a `pairs_count_{split}_{len}.txt` file. For each `reward_mode` in this rewards file, it counts the number of preference pairs that could be constructed from that rewward. This number should be passed into `eval_every` argument in DPO training and determines the number of samples per epoch.

See `scoring/confidence_scoring/compute_confidence_score_bio.sh` as an example for how to compute the confidence scores.

## DPO Training
The `factual_dpo` repo is a fork of DPO that takes in our FactTune truthfulness scores as input, converts them to preferences, and does DPO training.
``` 
git clone https://github.com/kttian/factual_dpo
```

See `factual_dpo/config/config.yaml` to see the set of default parameters used in DPO training. Under `data_prep`, you'll need to provide the paths of the score files generated in the previous section to `reward_file_train` and `reward_file_test`.

## Evaluation
Coming soon.

## Data Generation
Coming soon.

## Contact
If you have any questions or issues, please contact kattian@cs.stanford.edu.
