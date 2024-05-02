'''
Convert atomic facts to atomic questions. 1st step in computing AQ-based confidence scores.
'''
import argparse 
import json 
import os 
import openai 
import pathos.multiprocessing
import time 
import sys
import logging
import pytz
from datetime import datetime
from utils_atomic_q import get_convert_to_aq_prompt

sys.path.append('../../')
from generation import get_content_safe, estimate_cost


def atomic_fact_get_qs_single(args, name, topic_info, num_context=0):
    start_time = time.time()

    # input directory of factscore results
    fact_score_dir = os.path.join(args.DATA_DIR, 'factscore_outputs')
    
    # check if atomic questions output file already exits 
    aqs_file = os.path.join(args.aq_out_dir, f'{name}_aqs.json')
    print("aqs_file:", aqs_file)
    if os.path.exists(aqs_file):
        print(f"Already done w/ {name} since out file exists. Skipping!")
        logging.info(f"Already done w/ {name} since out file exists. Skipping!")
        return 
    
    # load fact score output file (input for atomic questions)
    fs_out_file = os.path.join(fact_score_dir, f'{name}_fs_out.json')
    if os.path.exists(fs_out_file):
        with open(fs_out_file, 'r') as f:
            fs_out_list = json.load(f)
        print(f"Loaded fact score output file {fs_out_file} with {len(fs_out_list)} responses.")
        logging.info(f"Loaded fact score output file {fs_out_file} with {len(fs_out_list)} responses.")
    else:
        print(f"ERROR: Factscore file does not exist ({fs_out_file}). Skip {name}.")
        logging.info(f"ERROR: Factscore file does not exist ({fs_out_file}). Skip {name}.")
        return 
    
    n = len(fs_out_list) # number of responses for this topic/name

    # estimate openai cost
    total_input_text = ""
    total_output_text = "" # use the atomic fact to estimate the length of the atomic question
    for i in range(n):
        decisions_list = fs_out_list[i]['decisions'][0]
        atomic_facts_list = [d['atom'] for d in decisions_list]
        for j in range(len(decisions_list)):
            prompt = get_convert_to_aq_prompt(name, atomic_facts_list[j], args.dataset, topic_info)
            if i == 0 and j == 0:
                print("Prompt:", len(prompt.split()))
            total_input_text += prompt 
            total_output_text += atomic_facts_list[j]
    estimate_cost(input_text=total_input_text, output_text=total_output_text)

    for i in range(n):
        # fs_out_list[i] has keys: 'topic', 'generation', 'score', 'respond_ratio', 
        # 'decisions', 'num_facts_per_response', 'init_score'

        # the response is broken down into the following atomic facts by factscore 
        decisions_list = fs_out_list[i]['decisions'][0]
        atomic_facts_list = [d['atom'] for d in decisions_list]
        atomic_questions_list = []
        # now, we convert each atomic fact into an atomic question
        for j in range(len(decisions_list)):
            # we are about to process response i and its atomic fact j for the given topic
            if i % 5 == 0 and j % 5 == 0:
                print(f"Processing response {i} and atomic fact {j} for topic {name}...")
            atomic_fact_j = atomic_facts_list[j]
            if num_context == 0:
                context_facts = []
                context_atomic_questions = []
            else:
                context_facts = atomic_facts_list[j-num_context:j] # previous atomic facts 
                context_atomic_questions = atomic_questions_list[j-num_context:j] # previous atomic questions 
            # get the GPT-3.5 prompt for converting the atomic fact to an atomic question
            prompt = get_convert_to_aq_prompt(name, atomic_fact_j, args.dataset, topic_info,
                                              context_facts, context_atomic_questions)
            # send the prompt to GPT-3.5 and get the atomic question
            new_atomic_q = get_content_safe(prompt, model=args.openai_model, seed=args.openai_seed)
            new_atomic_q = new_atomic_q.split("Statement:")[0].strip(" \n") # sometimes needed 
            atomic_questions_list.append(new_atomic_q)

            fs_out_list[i]['decisions'][0][j]['atomic_question'] = new_atomic_q
            fs_out_list[i]['decisions'][0][j]['convert_to_aq_prompt'] = prompt

    with open(aqs_file, 'w') as f:
        json.dump(fs_out_list, f, indent=4)
    print(f"Saved: {name} (time {time.time() - start_time:.3f})")
    logging.info(f"Saved: {name} (time {time.time() - start_time:.3f})")
    return


if __name__ == "__main__":
    ### arguments ### 
    parser = argparse.ArgumentParser()
    # args for running atomic_fact_get_qs_single
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--run_parallel', action='store_true')
    parser.add_argument('--num_cores', type=int, default=4)
    parser.add_argument('--num_context', type=int, default=0, help="number of context atomic facts for the prompt")
    parser.add_argument('--openai_key', type=str, default=None)
    parser.add_argument('--openai_model', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--openai_seed', type=int, default=0)
    # args for directory location 
    parser.add_argument('--DATA_DIR', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="bio", help="dataset type, e.g. bio/med")
    parser.add_argument('--dataset_loc', type=str, default=None, help="folder name for the dataset, in case there are different versions")
    parser.add_argument('--aq_out_dir', type=str, default=None)
    parser.add_argument('--split', type=str, required=True, help="train/dpoval/val/test. only need to score first 2")
    parser.add_argument('--questions_file', type=str, default=None)
    args = parser.parse_args()
    
    if args.dataset_loc is None:
        args.dataset_loc = args.dataset 
    if args.DATA_DIR is None:
        args.DATA_DIR = os.path.join(os.environ["FACT_TUNE_DIR"], "data", f"dataset_{args.dataset_loc}")
    if args.openai_key is None:
        args.openai_key = os.path.join(os.environ["FACT_TUNE_DIR"], "openai_key.txt")
    with open(args.openai_key, 'r') as f:
        openai_key = f.read().strip("\n ")
        openai.api_key = openai_key

    ### create output directory for this step ### 
    if args.aq_out_dir is None:
        args.aq_out_dir = os.path.join(args.DATA_DIR, "atomic_qs")
    if not os.path.exists(args.aq_out_dir):
        os.mkdir(args.aq_out_dir)
        print("Created directory:", args.aq_out_dir)

    ### logging ###
    now_id = datetime.now(pytz.timezone('US/Pacific')).strftime("%m%d-%H%M%S")
    log_dir = os.path.join(args.DATA_DIR, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = os.path.join(log_dir, f"atomic_q_{args.start}_{args.end}_{now_id}.log")
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(message)s')
    logging.info(args)
    print(args)

    ### load questions ###
    if args.questions_file is None:
        args.questions_file = os.path.join(args.DATA_DIR, f"questions_per_topic_{args.split}.json")
    with open(args.questions_file, 'r') as f:
        questions_list = json.load(f)
        print(f"Loaded questions_file {args.questions_file} with {len(questions_list)} questions.")

    ### convert questions_list to dict ###
    topic_info = {}
    for q_i in questions_list:
        topic_info[q_i['topic']] = q_i

    def map_fn(i):
        name = questions_list[i]["topic"]
        atomic_fact_get_qs_single(args, name, topic_info, args.num_context)
        return i
    
    args.end = min(args.end, len(questions_list))
    inputs = list(range(args.start, args.end))
    if args.debug:
        print(f"Running in debug mode from {args.start} to {args.end}...")
        for i in inputs:
            map_fn(i)
    elif not args.run_parallel: # run linearly
        print(f"Running linearly from {args.start} to {args.end}...")
        for i in inputs:
            name = questions_list[i]["topic"]
            print(f"\n{i} NAME: {name}")
            logging.info(f"\n{i} NAME: {name}")
            start_time = time.time()
            while True:
                try:
                    map_fn(i)
                    break 
                except Exception as e:
                    print(e)
                    print("Retrying...")
                    logging.info(f"Error: {e}")
                    logging.info("Retrying...")
                    aqs_file = os.path.join(args.DATA_DIR, "atomic_qs", f'{name}_aqs.json') 
                    if os.path.exists(aqs_file):
                        print(f"Removing {name} aqs output file...")
                        os.remove(aqs_file)
                    time.sleep(2)
            total_time = time.time() - start_time
            print(f"Time for {name}: {total_time:.4f}")
    else:
        # run with multiprocessing
        print(f"Running in parallel with {args.num_cores} cores...")
        while True:
            try:
                with pathos.multiprocessing.Pool(args.num_cores) as p:
                    responses = p.map(map_fn, inputs)
                    break 
            except Exception as e:
                print(e)
                print("Retrying...")
                time.sleep(1.5)
