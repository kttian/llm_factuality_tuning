import argparse 
from collections import defaultdict
import numpy as np 
import os 
import json 
from scipy.stats import entropy
import logging
import pytz
from datetime import datetime

import sys 
sys.path.append('../../')
from utils_score import chunk_list 
from parsing import normalize_sample, build_equiv_class_heuristics, answers_are_equivalent_heuristic

# function that takes in a set of samples and returns one reward of each type
def confidence_rewards(samples, original_ans=None):
    samples = [s for s in samples if s is not None] # remove None samples
    samples = [normalize_sample(s) for s in samples]
    if len(samples) == 0:
        return None
    equiv_classes = build_equiv_class_heuristics(samples, [1./len(samples)]*len(samples))
    confidences = list(equiv_classes.values())
    ent = entropy(confidences)
    max_conf = max(confidences)
    exp_conf = np.sum(np.array(confidences)**2)

    if original_ans is None:
        return {"neg_ent": -ent, "max_conf": max_conf, "exp_conf": exp_conf}
    
    original_ans = normalize_sample(original_ans)
    equiv_list = [answers_are_equivalent_heuristic(original_ans, s) for s in samples]
    samp_conf = np.mean(equiv_list)
    return {"neg_ent": -ent, "max_conf": max_conf, "samp_conf": samp_conf, "exp_conf": exp_conf}

'''
This script
- takes in all the factscore outputs and saved sampled answers to atomic units,
- computes confidence metrics on sampled answers
- saves one file of all rewards per response (confidence & factscore), ready for dpo. 
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="bio")
    parser.add_argument('--dataset_loc', type=str, default=None)
    parser.add_argument('--DATA_DIR', type=str, default=None)
    parser.add_argument('--aqs_folder_name', type=str, default="atomic_qs")
    parser.add_argument('--aq_samples_folder_name', type=str, default="atomic_qs_samples/llama-7b_20samples") #AQ
    parser.add_argument('--ne_samples_folder_name', type=str, default="ne_samples/llama7b_20samples") #NE
    parser.add_argument('--questions_file', type=str, default=None)
    parser.add_argument('--SAVE_DIR', type=str, default=None)
    parser.add_argument('--INCLUDE_NE', action='store_true')
    parser.add_argument('--num_resp_per_q', type=int, default=10) # max number of responses per question to use 
    parser.add_argument('--split', type=str, required=True)
    args = parser.parse_args()

    if args.dataset_loc is None:
        args.dataset_loc = args.dataset
    if args.DATA_DIR is None:
        args.DATA_DIR = os.path.join(os.environ['FACT_TUNE_DIR'], "data", f"dataset_{args.dataset_loc}")

    # logging
    now_id = datetime.now(pytz.timezone('US/Pacific')).strftime("%m%d-%H%M%S")
    log_dir = os.path.join(args.DATA_DIR, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = os.path.join(log_dir, f"metric_prep_rewards_{now_id}.log")
    print("LOG FILE:", log_file)
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(message)s')

    # input samples dir
    args.aqs_dir = os.path.join(args.DATA_DIR, args.aqs_folder_name)
    args.aq_samples_dir = os.path.join(args.DATA_DIR, args.aq_samples_folder_name)
    args.ne_samples_dir = os.path.join(args.DATA_DIR, args.ne_samples_folder_name)
    INCLUDE_NE = args.INCLUDE_NE and os.path.exists(args.ne_samples_dir)
    if not os.path.exists(args.ne_samples_dir):
        print(f"Warning: NE samples directory {args.ne_samples_dir} does not exist.")

    # rewards dir 
    if args.SAVE_DIR is None:
        args.SAVE_DIR = os.path.join(args.DATA_DIR, "rewards")

    # questions / topics file 
    if args.questions_file is None:
        args.questions_file = os.path.join(args.DATA_DIR, f"questions_per_topic_{args.split}.json")
    with open(args.questions_file, 'r') as f:
        questions_dict = json.load(f)
        print(f"Loaded questions_file {args.questions_file} with {len(questions_dict)} questions.")

    logging.info("ARGS:")
    logging.info(args)

    metrics_dict = {}
    '''
    metrics_dict will have the following structure:
    {
        "topic1": {
            "question1": {
                "responses": [R1, R2, R3, R4, R5],
                "fact_score": [0.8, 0.1, 0.1, 0.1, 0.1],
                "aq_max_conf": [0.8, 0.1, 0.1, 0.1, 0.1],
            }
        },
        "topic2": ...
    }
    '''
    
    resp_count = defaultdict(int)
    pairs_count = defaultdict(int)
    for t_i, data_point in enumerate(questions_dict):
        name = data_point["topic"]
        # load factscore / atomic questions output 
        fs_aq_outfile = os.path.join(args.aqs_dir, f"{name}_aqs.json")
        if not os.path.exists(fs_aq_outfile):
            print(f"Error ({t_i}): FS/AQ file {fs_aq_outfile} does not exist.")
            continue
        with open(fs_aq_outfile, 'r') as f:
            fs_aq_output = json.load(f)
        
        # load AQ sampled answers
        aq_samples_file = os.path.join(args.aq_samples_dir, f"{name}_samples.json")
        if not os.path.exists(aq_samples_file):
            print(f"Error ({t_i}): AQ samples file {aq_samples_file} does not exist.")
            continue
        with open(aq_samples_file, 'r') as f:
            aq_samples_dict = json.load(f)
        
        # load NE sampled answers 
        ne_samples_file = os.path.join(args.ne_samples_dir, f"{name}_ne_samples.json")
        if INCLUDE_NE:
            if not os.path.exists(ne_samples_file):
                print(f"Error ({t_i}): NE samples file {ne_samples_file} does not exist.")
                continue
            with open(ne_samples_file, 'r') as f:
                ne_samples_dict = json.load(f)

        questions = data_point["questions"]
        n_q = len(questions)
        n = len(fs_aq_output)//n_q # number of responses per question 
        fs_aq_output_chunk_list = chunk_list(fs_aq_output, n_q, n)

        metrics_dict[name] = {}
        for q_i, q in enumerate(questions):
            rewards_dict = defaultdict(list)
            fs_aq_per_response = fs_aq_output_chunk_list[q_i][:args.num_resp_per_q] # len n
            rewards_dict["responses"] = [x["generation"] for x in fs_aq_per_response]

            # factscore
            rewards_dict["factscore_avg"] = [x["init_score"] for x in fs_aq_per_response]
            rewards_dict["factscore_lp_avg"] = [x["score"] for x in fs_aq_per_response]

            # confidence-based scores 
            for r_j, fs_aq_out in enumerate(fs_aq_per_response):
                resp = fs_aq_out["generation"]
                # AQ rewards 
                conf_in_response = defaultdict(list)
                for decision in fs_aq_out['decisions'][0]:
                    atomic_question_k = decision["atomic_question"]
                    aq_samples = aq_samples_dict['samples'][q_i*n+r_j][atomic_question_k]
                    rd = confidence_rewards(aq_samples, None)
                    for key in rd: # neg_ent, max_conf, samp_conf
                        conf_in_response[f"aq_{key}"].append(rd[key])
            
                # NE rewards
                if INCLUDE_NE:
                    ne_dict = ne_samples_dict[q][resp]
                    for ent in ne_dict:
                        ne_samples = ne_dict[ent]
                        rd = confidence_rewards(ne_samples, None)
                        for key in rd:
                            conf_in_response[f"ne_{key}"].append(rd[key])
                
                for key in conf_in_response:
                    rewards_dict[f"{key}_avg"].append(np.nanmean(conf_in_response[key]))
            
            # counting dpo pairs
            reward_diff_thresh = 0.
            for key in rewards_dict:
                if key == "responses":
                    continue
                rewards = rewards_dict[key]
                n = len(rewards)
                resp_count[key] += n

                for i in range(n):
                    for j in range(i+1, n):
                        if rewards[i] > rewards[j] + reward_diff_thresh:
                            pairs_count[key] += 1
                        elif rewards[j] > rewards[i] + reward_diff_thresh:
                            pairs_count[key] += 1

            metrics_dict[name][q] = rewards_dict
    ### end of loop 

    l = len(metrics_dict)
    save_file = os.path.join(args.SAVE_DIR, f"rewards_{args.split}_{l}.json")
    logging.info(f"Saving all rewards (l={l}) to {save_file}")
    with open(save_file, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    count_file = os.path.join(args.SAVE_DIR, f"pairs_count_{args.split}_{l}.txt")
    for key in pairs_count:
        print(f"{key}:  \t{pairs_count[key]}")
        with open(count_file, 'w') as f:
            f.write(f"{key}:  \t{pairs_count[key]}\n")
        logging.info(f"{key}:  \t{pairs_count[key]}")
