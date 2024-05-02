import argparse
import json 
import time 
import os 
import logging 
import socket 
import pytz
from datetime import datetime

from factscore.factscorer import FactScorer


def factscore_single(args, data_point):
    '''
    Example data_point:
    {
        "topic": "Acne",
        "topic_id": 1,
        "questions": [
            "What causes acne?",
            "How can acne be treated?",
            "What are the types of acne?"
        ]
    }

    Example response_dict:
    {
        "What causes acne?": [
            "Acne is most commonly caused by a combination of hormones and the skin's overproduction of oil, called sebum, which clogs the pores. It can also be caused by bacteria, called P. acnes, that naturally occur on the skin, but when too much sebum or other irritants clog the pores, this bacteria can multiply and lead to acne. The good news is that acne is treatable, and most acne can be controlled or even eliminated. Treatments typically include topical (applied to the skin) medications, medications to be taken by mouth, and, in some cases, procedures to remove clogged pores. For more information on treating acne, consult your dermatologist.",
            "Acne is an inflammatory condition of the skin that is triggered when hair follicles become plugged with oil and dead skin cells. Oil and dirt build up under the skin, creating an environment in which bacteria can thrive. This can cause the development of pimples, blackheads, whiteheads, cysts and other lesions that can be painful and may lead to scarring. The most common cause of acne is hormonal changes in adolescence. Hormones are responsible for producing oil from the sebaceous glands and this, in turn, leads to clogged pores and acne. In some cases, adult women may also experience hormonal changes in their 20s or 30s that may trigger acne. This is known as adult acne or hormonal acne.",
            ...
        ],
        "How can acne be treated?": [
            "It is important to have a dermatologist diagnose and properly treat acne. There are many different treatments, and they will depend on the severity of the acne and the patient's skin type. People with oily skin should avoid products with harsh ingredients like salicylic acid or alcohol. They should also use a gentle skin cleanser and avoid wearing tight clothing. People who have acne on the face may want to try over-the-counter creams or lotions, but it is important to talk to a dermatologist before using these products.",
            "Many people are plagued by this condition, as acne affects up to 50 million Americans each year. This skin disease can cause painful bumps and spots that leave the skin looking inflamed, and it usually affects teens. The best treatment for acne is based on the type of acne that the patient has. Acne can be classified as inflammatory or non-inflammatory, and both types can be treated with medication. For non-inflammatory acne, over-the-counter products can be used to remove the excess oil and debris from the face. For those with acne that is more severe, oral antibiotics and topical medication can be used in combination with oral contraceptives or birth control pills. This can help reduce the number of new blemishes.",
            ...
        ],
        ...
    }
    '''
    start_time = time.time()

    topic = data_point["topic"]
    topic_id = data_point["topic_id"]
    questions = data_point["questions"]

    fs_out_file = os.path.join(args.fs_out_dir, f"{topic}_fs_out.json")
    if not args.debug and os.path.exists(fs_out_file):
        print(f"Skipping {topic} ({topic_id}) - out file already exists!")
        logging.info(f"Skipping {topic} ({topic_id}) - out file already exists!")
        return 

    response_file = os.path.join(args.responses_dir, f"{topic}_generations.json")
    if os.path.exists(response_file):
        with open(response_file, 'r') as f:
            responses_dict = json.load(f)
        print(f"Loaded {len(responses_dict)} questions for {topic}.")
        logging.info(f"Loaded {len(responses_dict)} questions for {topic}.")
    else:
        print(f"ERROR: Responses file does not exist ({response_file}). Skip {topic}.")
        logging.info(f"ERROR: Responses file does not exist ({response_file}). Skip {topic}.")
        return  

    def flatten_responses_dict(responses_dict):
        responses = []
        for question in questions:
            # responses.extend(responses_dict[question])
            responses.extend(responses_dict[question]["responses"])
        return responses

    def unflatten_responses_dict(responses, questions):
        num_generations = len(responses) / len(questions)
        if num_generations != int(num_generations):
            print("ERROR: num_generations not an integer. Exit.")
            return
        num_generations = int(num_generations)
        responses_dict = {}
        for i, question in enumerate(questions):
            responses_dict[question] = responses[i*num_generations:(i+1)*num_generations]
        return responses_dict
    
    responses = flatten_responses_dict(responses_dict)

    def process_fs_out_list(fs_out_list):
        # convert np.bool to bool so that we can save as json 
        for i in range(len(fs_out_list)):
            fs_out = fs_out_list[i]
            for j in range(len(fs_out['decisions'][0])):
                fs_out['decisions'][0][j]['is_supported'] = bool(fs_out['decisions'][0][j]['is_supported'])
            fs_out_list[i] = fs_out
        return fs_out_list

    def compute_fs_out_list():
        # call factscoring function for all responses and saves the output
        fs_out_list = fs.get_score([topic]*len(responses), responses, chunk_size=0, is_bio=True)
        fs_out_list = process_fs_out_list(fs_out_list)
        with open(fs_out_file, 'w') as f:
            json.dump(fs_out_list, f, indent = 4)
        return fs_out_list
        
    if args.debug:
        print("RESPONSES:", len(responses))
        fs_out_list = compute_fs_out_list()
        print("FACT SCORE OUTPUTS:")
        print(fs_out_list)
        duration = time.time() - start_time
        print(f"Time: {duration:.4f} s")
        logging.info(f"Time: {duration:.4f} s")

    else:
        try:
            compute_fs_out_list()
            duration = time.time() - start_time
            print(f"Time: {duration:.4f} s")
            logging.info(f"Time: {duration:.4f} s")
        except:
            print(f"FACTSCORE ERROR: topic {topic} ({topic_id}). SKIP FOR NOW.")
            logging.info(f"FACTSCORE ERROR: topic {topic} ({topic_id}). SKIP FOR NOW.")
            if os.path.exists(fs_out_file):
                os.remove(fs_out_file) # remove the file if it's incomplete
    return 


if __name__ == "__main__":
    '''
    Given a dataset of prompts and responses, this script computes the factscore each response.

    Dataset Format: Dataset has N prompts and n responses per prompt. Each prompt also has a "topic", 
    or wikipedia article title, to use as the reference in factscore. The dataset is saved as a 
    - questions file (default: $DATA_DIR/questions_per_topic_$SPLIT.json) and 
    - a directory of responses (default: $DATA_DIR/responses/$TOPIC_generations.json).

    factscore_single takes in a topic id and scores all the responses for that topic.

    Then, this script runs factscore_single in a loop or in parallel.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--debug', action='store_true')
    # factscore arguments
    # please see https://github.com/kttian/FActScore/blob/main/README.md for more details 
    parser.add_argument('--fs_model_mode', type=str, default="retrieval+llama+npm",
                        help="Model for factscore: retrieval+llama+npm or retrieval+ChatGPT")
    parser.add_argument('--fs_cache_base', type=str, default=os.environ["FS_CACHE_BASE"],
                        help="Path to .cache directory for FactScore")
    parser.add_argument('--fs_save_folder', type=str, default="factscore", 
                        help="Subfolder of fs_cache_base, where FactScore model weights and data are loaded")
    parser.add_argument('--fs_cache_folder', type=str, default="factscore", 
                        help="Cache subfolder of fs_cache_base. 2 runs with the same cache dir may cause problems")
    parser.add_argument('--oai_key', type=str, default=os.path.join(os.environ["FACT_TUNE_DIR"], "openai_key.txt"), 
                        help="Path to .txt containing openai key")
    parser.add_argument('--oai_org', type=str, default="", help="openai org id")
    parser.add_argument('--af_model_name', type=str, default="ChatGPT", help="Atomic Facts model name")
    # dataset arguments
    parser.add_argument('--dataset_loc', type=str, default="bio")
    parser.add_argument('--DATA_DIR', type=str, default=None,
                        help="Parent directory containing dataset and where we will save intermediate outputs")
    parser.add_argument('--questions_file', type=str, default=None, help="File containing a list of questions per topic")
    parser.add_argument('--responses_dir', type=str, default=None, help="Dir where each file contains the responses for each topic")
    parser.add_argument('--fs_out_dir', type=str, default=None, help="Dir to save factscore outputs per topic")
    parser.add_argument('--split', type=str, required=True, help="train, dpoval, val or test. Only need to score first 2")
    args = parser.parse_args()
    print(args)

    if args.DATA_DIR is None:
        args.DATA_DIR = os.path.join(os.environ["FACT_TUNE_DIR"], "data", f"dataset_{args.dataset_loc}")

    now_id = datetime.now(pytz.timezone('US/Pacific')).strftime("%m%d-%H%M%S")
    log_dir = os.path.join(args.DATA_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(args.DATA_DIR, "logs", f"factscore_st{args.start}_en{args.end}_{now_id}.log")
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(message)s')

    # get hostname
    hostname = socket.gethostname()
    logging.info(f"Running on: {hostname}")

    # we always save the factscore model weights and data in the same "save_dir" across runs
    # however, we need separate "fs_cache_dir" for simulateous runs on the same machine
    fs_save_dir = os.path.join(args.fs_cache_base, args.fs_save_folder)
    fs_cache_dir = os.path.join(args.fs_cache_base, args.fs_cache_folder)
    print("Factscore Model/Data Save Dir:", fs_save_dir)
    print("Factscore Cache Dir:", fs_cache_dir)
    if not os.path.exists(fs_cache_dir):
        os.makedirs(fs_cache_dir)
        print("made dir", fs_cache_dir)
    

    # create factscorer object
    fs = FactScorer(model_name=args.fs_model_mode, openai_key=args.oai_key, openai_org=args.oai_org, 
                    af_model_name=args.af_model_name,
                    data_dir=fs_save_dir, model_dir=fs_save_dir, cache_dir=fs_cache_dir, verbose=True)
    

    ### load questions ###
    if args.questions_file is None:
        args.questions_file = os.path.join(args.DATA_DIR, f"questions_per_topic_{args.split}.json")
    with open(args.questions_file, 'r') as f:
        questions_dict = json.load(f)
    
    if args.responses_dir is None:
        args.responses_dir = os.path.join(args.DATA_DIR, "responses")
    
    ### create factscore outputs directory ###
    if args.fs_out_dir is None:
        args.fs_out_dir = os.path.join(args.DATA_DIR, "factscore_outputs")
    if not os.path.exists(args.fs_out_dir):
        os.makedirs(args.fs_out_dir)
        print("made fs out dir", args.fs_out_dir)
    
    # print questions path, responses path, fs_out path
    print("Questions File:", args.questions_file)
    print("Responses Dir:", args.responses_dir)
    print("Factscore Outputs Dir:", args.fs_out_dir)
    logging.info(f"Questions File: {args.questions_file}")
    logging.info(f"Responses Dir: {args.responses_dir}")
    logging.info(f"Factscore Outputs Dir: {args.fs_out_dir}")

    #######################################################################
    ### loop through topics & run factscore for each question per topic ###
    args.end = min(args.end, len(questions_dict))
    for idx in range(args.start, args.end):
        one_topic = questions_dict[idx]
        print()
        print("="*100)
        print(idx, one_topic)
        print()

        factscore_single(args, one_topic)
        if idx % 10 == 0:
            print("Sleeping for 2 seconds (safe to exit)...")
            time.sleep(2) # need to sleep once in a while for safe quit 
            print("End sleep!")
