import argparse 
import json 
import numpy as np 
import os 
import random 
import socket
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch 
import time 
import tqdm
import logging
import pytz
from datetime import datetime
import getpass

from utils_atomic_sample import get_aq_sampling_prompt

import sys
sys.path.append("../../")
from utils import load_model_tok


def get_generations(prompts, chunk_size=4, temperature=1.0, samples_per_prompt=1, top_p=0.9, max_new_tokens=15):
    prompts = [p for p in prompts for _ in range(samples_per_prompt)]
    responses = []
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    for i in tqdm.tqdm(list(range(0, len(prompts), chunk_size))):
        chunk = prompts[i:i+chunk_size]
        inputs = tokenizer(chunk, return_tensors='pt', padding=True) #add_special_tokens=False, 
        input_len = inputs['input_ids'].shape[1]
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, 
                                 num_return_sequences=1, top_p=top_p, pad_token_id=tokenizer.eos_token_id)
        outputs = tokenizer.batch_decode(outputs[:,input_len:], skip_special_tokens=True)
        # print("outputs:", outputs)
        responses.extend(outputs)
    return responses 

def aq_sampling(name, args):
    real_start_time = time.time()
    # load questions file, then sample answers from llama
    samples_file = os.path.join(args.samples_dir, f'{name}_samples.json')
    if os.path.exists(samples_file):
        print("Already done:", name)
        logging.info(f"Already done with {name} since file exists. Skip!")
        return
    
    qs_file = os.path.join(args.aq_dir, f'{name}_aqs.json')
    if os.path.exists(qs_file):
        with open(qs_file, 'r') as f:
            fs_out_list = json.load(f)
    else:
        print(f"FS/AQ input file not found for {name}. Skip!")
        logging.info(f"FS/AQ input file not found for {name}. Skip!")
        return
    
    n = len(fs_out_list)
    print("# of Paragraphs:", n)
    all_samples = [] 
    _all_questions = [] 
    qs_dict = {}
    for i in range(n):
        # for each paragraph, create a atomic question-to-samples dict 
        # (since there can be overlap of aq's between responses)
        atomic_questions_list = [fs_out_list[i]['decisions'][0][j]['atomic_question'] 
                                for j in range(len(fs_out_list[i]['decisions'][0]))]
        _all_questions.extend(atomic_questions_list)
        # atomic_questions_list = list(set(atomic_questions_list))
        # instead of using a set, using a list keeps the order of the questions fixed
        atomic_questions_list_unique = []
        for q in atomic_questions_list:
            if q not in atomic_questions_list_unique:
                atomic_questions_list_unique.append(q)
        atomic_questions_list = atomic_questions_list_unique
        print(f"# atomic questions for paragraph {i}:", len(atomic_questions_list))

        prompts_list = [get_aq_sampling_prompt(q, args.dataset) for q in atomic_questions_list]
        if i == 0:
            print("example QUESTION:", atomic_questions_list[0])
            print("example PROMPT:", repr(prompts_list[0]))

        responses = get_generations(prompts_list, chunk_size=args.chunk_size, 
                                    samples_per_prompt=args.num_samples, max_new_tokens=15)
        samples_dict = {}
        for q_i, question in enumerate(atomic_questions_list):
            samples = responses[q_i*args.num_samples:(q_i+1)*args.num_samples]
            samples = [s.split("\n")[0] for s in samples]
            samples_dict[question] = samples
        if i == 0:
            print("example SAMPLES:", samples)
        all_samples.append(samples_dict)
    print(f"all_questions len: {len(_all_questions)}")
    _all_questions_set = list(set(_all_questions))
    print(f"all_questions_set len: {len(_all_questions_set)}")
        # total_time = time.time() - start_time
        # print(f"Time for paragraph {i}: {total_time:.4f}")
    
    qs_dict["samples"] = all_samples
    print(qs_dict.keys())
    with open(samples_file, 'w') as f:
        json.dump(qs_dict, f, indent=4)
    print("Saved:", name)
    print(f"Time for all paragraphs {name}:", time.time() - real_start_time)
    logging.info(f"Time for all paragraphs {name}: {time.time() - real_start_time:.4f}")


if __name__ == "__main__":
    ### arguments ### 
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=400)
    # sampling args 
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    # model args
    parser.add_argument('--model_name', type=str, default="llama-7b") # model for sampling
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--model_dtype', type=str, default='float16')
    parser.add_argument('--num_bits', type=int, default=16)
    parser.add_argument('--max_new_tok', type=int, default=15) # for usually simple questions
    parser.add_argument('--chunk_size', type=int, default=20) # 20 on 48GB, 40 on 80GB
    parser.add_argument('--cache_dir', type=str, default=f"/scr/{getpass.getuser()}")
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--hf_auth_token_path', type=str, default=None)
    # directory args
    parser.add_argument('--DATA_DIR', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="bio", help="dataset type, e.g. bio/med")
    parser.add_argument('--dataset_loc', type=str, default=None, help="folder name for the dataset, in case there are different versions")
    parser.add_argument('--aq_dir', type=str, default=None)
    parser.add_argument('--samples_parent_dir', type=str, default=None)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--questions_file', type=str, default=None)
    args = parser.parse_args()
    overall_start_time = time.time()

    print(f"Host name: {socket.gethostname()}")
    print(args)

    if args.dataset_loc is None:
        args.dataset_loc = args.dataset 
    if args.DATA_DIR is None:
        args.DATA_DIR = os.path.join(os.environ["FACT_TUNE_DIR"], "data", f"dataset_{args.dataset_loc}")
    # input directory of atomic questions
    if args.aq_dir is None:
        args.aq_dir = os.path.join(args.DATA_DIR, "atomic_qs")
    # read huggingface auth token
    if args.hf_auth_token_path:
        with open(args.hf_auth_token_path, 'r') as f:
            args.auth_token = f.read().strip()
    else:
        args.auth_token = None
    
    ### create output directory for this step ### 
    if args.samples_parent_dir is None:
        args.samples_parent_dir = os.path.join(args.DATA_DIR, "atomic_qs_samples")
    if not os.path.exists(args.samples_parent_dir):
        os.mkdir(args.samples_parent_dir)
        print("Created directory:", args.samples_parent_dir)
    folder_name = args.model_name
    if args.model_dtype != "float16":
        folder_name += f"_{args.model_dtype}"
    if args.num_bits != 16:
        folder_name += f"_{args.num_bits}bit"
    folder_name += f"_{args.num_samples}samples"
    args.samples_dir = os.path.join(args.samples_parent_dir, folder_name)
    print("SAMPLES DIR:", args.samples_dir)
    if not os.path.exists(args.samples_dir):
        os.makedirs(args.samples_dir)
        print("Created directory:", args.samples_dir)
    
    now_id = datetime.now(pytz.timezone('US/Pacific')).strftime("%m%d-%H%M%S")
    log_file = os.path.join(args.DATA_DIR, "logs", f"sample_answers_{args.start}_{args.end}_{now_id}.log")
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(message)s')
    logging.info(f'Host name: {socket.gethostname()}')
    logging.info(args)

    ### load model and tokenizer ###
    model, tokenizer = load_model_tok(args)
    model.eval()
    
    ### load questions ###
    if args.questions_file is None:
        args.questions_file = os.path.join(args.DATA_DIR, f"questions_per_topic_{args.split}.json")
    with open(args.questions_file, 'r') as f:
        questions_list = json.load(f)
        print(f"Loaded questions_file {args.questions_file} with {len(questions_list)} questions.")

    args.end = min(args.end, len(questions_list))
    inputs = list(range(args.start, args.end))
    prev_err_id = -100
    for i in inputs:
        name = questions_list[i]["topic"]
        print(f"Starting {i} - {name}")
        logging.info(f"Starting {i} - {name}")
        try:
            aq_sampling(name, args)
        except Exception as e:
            print("name:", name, " - ERROR:", e)
            if i == prev_err_id + 1:
                print("Error twice in a row -- stopping.")
                exit() 
            else:
                prev_err_id = i 
                continue
    
    overall_total_time = time.time() - overall_start_time 
    print("Total time:", overall_total_time)
