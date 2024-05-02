import argparse
import json
import numpy as np
import os
import spacy
nlp = spacy.load("en_core_web_sm")
import socket
import random
import time
import torch
from   datetime import datetime
import logging
import pytz
import getpass 

import sys
sys.path.append("../../")
from utils import load_model_tok

def extract_entities(response):
    # for medical qa, use noun chunks as a proxy for important entities 
    excluded_noun_chunks = ["which", "that", "it", "this"]
    doc = nlp(response)
    entities = []
    for chunk in doc.noun_chunks:
        # filter out noun chunks that are subjects, since there is too much open-ended uncertainty 
        if chunk.root.dep_ != "nsubj" and chunk.root.dep_ != "nsubjpass" and chunk.root.dep_ != "ROOT":
            if chunk.text.strip().lower() not in excluded_noun_chunks:
                # print(chunk.text, ":", chunk.start_char, chunk.end_char, chunk.start, chunk.label_)
                entities.append(chunk)
    return entities 

def ne_sampling_single(name):
    start_time = time.time()
    samples_file = os.path.join(args.samples_dir, f'{name}_ne_samples.json')
    if os.path.exists(samples_file):
        print("Already done:", name)
        logging.info(f"Already done with {name} since file exists. Skip!")
        return
    
    response_file = os.path.join(args.responses_dir, f"{name}_generations.json")
    if os.path.exists(response_file):
        with open(response_file, 'r') as f:
            responses_dict = json.load(f)
        print(f"Loaded {len(responses_dict)} questions for {name}.")
        logging.info(f"Loaded {len(responses_dict)} questions for {name}.")
    else:
        print(f"ERROR: Responses file does not exist ({response_file}). Skip {name}.")
        logging.info(f"ERROR: Responses file does not exist ({response_file}). Skip {name}.")
        return
    
    # samples_dict output format:
    # for each question, for each response, for each extracted entity, save a list of 20 (etc.) samples 
    samples_dict = {}
    # loop through questions
    for q in responses_dict:
        samples_dict[q] = {} # set up samples_dict 
        # loop through responses 
        for r_i in range(len(responses_dict[q]["responses"])):
            response = responses_dict[q]["responses"][r_i]
            samples_dict[q][response] = {} # set up samples_dict 
            text = q + " " + response
            entities = extract_entities(response)
            
            for j, ent in enumerate(entities):
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                random.seed(args.seed)
                start_char = ent.start_char + len(q) + 1
                # encode text prefix (text[:start_char-1]) before this entity. -1 to remove trailing space 
                prefix_text = text[:start_char-1]
                input_tokens = tokenizer.encode(prefix_text, return_tensors="pt", padding=False, truncation=True, max_length=512)
                input_tokens = input_tokens.cuda()
                entity_tokens = tokenizer.encode(ent.text, add_special_tokens=False, padding=False, truncation=True, max_length=512)
                
                model_out = model.generate(input_tokens, max_new_tokens=len(entity_tokens), top_p=args.top_p, do_sample=True, num_return_sequences=args.num_samples)
                samples_list = tokenizer.batch_decode(model_out[:, input_tokens.shape[1]:], skip_special_tokens=True)
                samples_dict[q][response][ent.text] = samples_list # set up samples_dict

    with open(samples_file, 'w') as f:
        json.dump(samples_dict, f, indent=4)
    print(f"Finished {name} in time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=200)
    # sampling args 
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    # model args 
    parser.add_argument('--model_name', type=str, default="llama7b")
    parser.add_argument('--cache_dir', type=str, default=f"/scr/{getpass.getuser()}")
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--top_p', type=int, default=0.9)
    parser.add_argument('--temperature', type=int, default=1.0)
    parser.add_argument('--model_dtype', type=str, default='float16')
    parser.add_argument('--num_bits', type=int, default=16)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--hf_auth_token_path', type=str, default=None)
    # directory args
    parser.add_argument('--DATA_DIR', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="bio")
    parser.add_argument('--dataset_loc', type=str, default=None)  
    parser.add_argument('--responses_dir', type=str, default=None) # input directory -- raw responses 
    parser.add_argument('--samples_parent_dir', type=str, default=None) # output directory -- NE samples
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--questions_file', type=str, default=None)
    args = parser.parse_args()

    ### input directory ###
    if args.dataset_loc is None:
        args.dataset_loc = args.dataset  
    if args.DATA_DIR is None:
        args.DATA_DIR = os.path.join(os.environ["FACT_TUNE_DIR"], "data", f"dataset_{args.dataset_loc}")
    if args.responses_dir is None:
        args.responses_dir = os.path.join(args.DATA_DIR, "responses")
    # read huggingface auth token
    if args.hf_auth_token_path:
        with open(args.hf_auth_token_path, 'r') as f:
            args.auth_token = f.read().strip()
    else:
        args.auth_token = None

    ### set up logger ###
    now_id = datetime.now(pytz.timezone('US/Pacific')).strftime("%m%d-%H%M%S")
    log_file = os.path.join(args.DATA_DIR, "logs", f"ne_sample_{args.start}_{args.end}_{now_id}.log")
    print("log path:", log_file)
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(message)s')
    logging.info(f'Host name: {socket.gethostname()}')

    ### create output directory for this step (NE Samples) ### 
    if args.samples_parent_dir is None:
        args.samples_parent_dir = os.path.join(args.DATA_DIR, "ne_samples")
    if not os.path.exists(args.samples_parent_dir):
        os.mkdir(args.samples_parent_dir)
        print("Created parent directory:", args.samples_parent_dir)
    # create sub directory for each model used to sample 
    folder_name = args.model_name
    if args.model_dtype != "float16":
        folder_name += f"_{args.model_dtype}"
    if args.num_bits != 16:
        folder_name += f"_{args.num_bits}bit"
    folder_name += f"_{args.num_samples}samples"
    args.samples_dir = os.path.join(args.samples_parent_dir, folder_name)
    if not os.path.exists(args.samples_dir):
        os.makedirs(args.samples_dir)
        print("Created args.samples_dir:", args.samples_dir)
        logging.info(f"Created args.samples_dir: {args.samples_dir}")
    else:
        print("args.samples_dir already exists:", args.samples_dir)
        logging.info(f"args.samples_dir already exists: {args.samples_dir}")
    logging.info("ARGS:")
    logging.info(args)

    ### load model and tokenizer ###
    model, tokenizer = load_model_tok(args)
    model.eval()

    ### load questions ###
    if args.questions_file is None:
        args.questions_file = os.path.join(args.DATA_DIR, f"questions_per_topic_{args.split}.json")
    with open(args.questions_file, 'r') as f:
        questions_dict = json.load(f)
        print(f"Loaded questions_file {args.questions_file} with {len(questions_dict)} questions.")
        logging.info(f"Loaded questions_file {args.questions_file} with {len(questions_dict)} questions.")
    
    args.end = min(args.end, len(questions_dict))
    inputs = list(range(args.start, args.end))
    
    for i in range(args.start, args.end):
        name = questions_dict[i]["topic"]
        print(f"Starting {i} - {name}")
        logging.info(f"Starting {i} - {name}")
        ne_sampling_single(name)
