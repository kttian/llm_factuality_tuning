import os
import hashlib
import json
import openai
import time 
import sys
import logging
import numpy as np 
from parsing import normalize_ans


NFS_CACHE_DIR = os.environ["NFS_OPENAI_CACHE"]
# open ai system prompt; can be different from the llama2 system prompt in the .txt
DEFAULT_SYSTEM_PROMPT = "You are an intelligent, honest, and harmless assistant. Your direct, concise responses contain only the minimum words needed to convey the answer."


def _cached_function(fn_to_cache, cache_dir=NFS_CACHE_DIR, rerun=False):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    def wrapped(*args, **kwargs):
        json_dump_args_kwargs = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
        hash = hashlib.sha256(json_dump_args_kwargs.encode('utf-8')).hexdigest()
        cache_path = os.path.join(cache_dir, hash)
        if not rerun and os.path.exists(cache_path):
            # print("load cache")
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                pass 
                # if exception, also want to rerun, so we continue... 
        result = fn_to_cache(*args, **kwargs)
        with open(cache_path, 'w') as f:
            json.dump(result, f)
        return result

    return wrapped


_openai_chat_completion = _cached_function(openai.ChatCompletion.create, rerun=False)


def get_response(prompt, model="gpt-3.5-turbo", system_prompt=DEFAULT_SYSTEM_PROMPT, seed=None):
    if system_prompt is None:
        messages = [
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    if seed is None:
        response = _openai_chat_completion(
            model=model,
            messages=messages
        )
    else:
        response = _openai_chat_completion(
            model=model,
            messages=messages,
            seed=seed
        )
    return response 


def get_content(prompt, model="gpt-3.5-turbo", system_prompt=DEFAULT_SYSTEM_PROMPT, seed=None, openai_key=None, openai_org=None): 
    response = get_response(prompt, model=model, system_prompt=system_prompt, seed=seed)
    return response["choices"][0]["message"]["content"]


def estimate_cost(input_text, output_text=None, model="gpt-3.5-turbo"):
    if model == "gpt-3.5-turbo":    
        input_cost = 0.0005 # $0.06 per 1K tokens 
        output_cost = 0.0015
    elif model == "gpt-3.5-turbo-instruct":
        input_cost = 0.0015
        output_cost = 0.0020
    elif model == "gpt-4":
        input_cost = 0.03
        output_cost = 0.06
    else:
        raise ValueError(f"Unknown model: {model}")

    # compute the estimated cost
    input_num_words = len(input_text.split())
    input_num_tokens = 4./3.*input_num_words
    input_cost = input_cost*input_num_tokens/1000.

    if output_text is not None:
        output_num_words = len(output_text.split())
        output_num_tokens = 4./3.*output_num_words
        output_cost = output_cost*output_num_tokens/1000.
    else:
        output_cost = 0.0
        output_num_tokens = 0

    total_cost = input_cost + output_cost

    print(f"Cost Estimate:")
    print(f"Prompt length: {input_num_tokens:.1f} input tokens, {output_num_tokens:.1f} output tokens")
    print(f"Total cost: ${total_cost:.5f} | Input cost: ${input_cost:.5f}, Output cost: ${output_cost:.5f}\n")

    return {"input_cost": input_cost, "output_cost": output_cost, "total_cost": total_cost,
            "input_num_tokens": input_num_tokens, "output_num_tokens": output_num_tokens}


def get_content_safe(prompt, model="gpt-3.5-turbo", system_prompt=DEFAULT_SYSTEM_PROMPT, seed=None, openai_key=None, openai_org=None): 
    received = False
    num_rate_errors = 0
    content = None

    while not received:
        try:
            response = get_response(prompt, model=model, system_prompt=system_prompt, seed=seed)
            content = response["choices"][0]["message"]["content"]
            received = True
        except:
            # print(message)
            num_rate_errors += 1
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            
            logging.error("API error: %s (%d). Waiting %.2f sec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))
    return content


def check_basic(a,b):
    # returns false if the answers a and b are obviously not equivalent
    if a == "yes" and b == "no":
        return False 
    if a == "no" and b == "yes":
        return False
    if a == "yes" and b == "sometimes":
        return False
    if a == "sometimes" and b == "yes":
        return False
    if a == "no" and b == "sometimes":
        return False
    if b == "sometimes" and a == "no":
        return False
    return True


def answers_to_q_are_equivalent_llm(question, a, b, model='gpt-3.5-turbo', verbose=False):
    if a == "" or b == "":
        return False
    if a == b:
        return True 
    if not check_basic(a,b):
        return False
    
    prompt = f'Are the following two answers to my question Q semantically equivalent?\n\nQ: {question}\nA1: {a}\nA2: {b}\n\nPlease answer with a single word, either "Yes." or "No.", and explain your reasoning.'
    response = _openai_chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    if verbose:
        print(response['choices'][0]['message']['content'])
    
    answer = response['choices'][0]['message']['content'].strip().lower() # answer + explanation 
    answer = normalize_ans(answer)
    answer = answer.split(".")[0] # extract yes/no answer from answer + explanation 
    if answer[-3:] == "yes" or answer[:3] == "yes":
        answer = "yes"
    elif answer[-2:] == "no" or answer[:2] == "no":
        answer = "no"
    if answer not in ['yes', 'no']:
        print(f'WARNING: unexpected answer from equivalence LLM: "{answer}"\nQuestion: "{question} \na: {a} \nb: {b} \n"')
    return answer == 'yes' 


def responses_are_equivalent_llm(a, b, model='gpt-3.5-turbo', verbose=False):
    if a == "" or b == "":
        return False
    if a == b:
        return True 
    if not check_basic(a,b):
        return False
    
    prompt = f'Are the following two candidate responses semantically equivalent?\n\n1. {a}\n2. {b}\n\nPlease answer with a single word, either "Yes." or "No.", and explain your reasoning.'
    response = _openai_chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    if verbose:
        print(response['choices'][0]['message']['content'])
    
    answer = response['choices'][0]['message']['content'].strip().lower() # answer + explanation 
    answer = normalize_ans(answer)
    answer = answer.split(".")[0] # extract yes/no answer from answer + explanation 
    if answer[-3:] == "yes" or answer[:3] == "yes":
        answer = "yes"
    elif answer[-2:] == "no" or answer[:2] == "no":
        answer = "no"
    if answer not in ['yes', 'no']:
        print(f'WARNING: unexpected answer from equivalence LLM: "{answer}"\na: {a} \nb: {b} \n"')
    return answer == 'yes' 


def add_to_equiv_class_llm(equiv_classes, ans, conf, question, model='gpt-3.5-turbo', max_comp=3):
    # assume ans already normalized
    keys_list = list(equiv_classes.keys())
    comps = 0
    for key in keys_list[:max_comp]:
        sleep_time = 1.5
        while True:
            try:
                equiv_bool = answers_to_q_are_equivalent_llm(question, ans, key, model=model)
                break
            except Exception as e:
                print(e)
                print(f"Retrying... sleep {sleep_time}")
                time.sleep(sleep_time)
                sleep_time *= 1.8

        comps += 1
        if equiv_bool:
            equiv_classes[key] += conf
            return comps 
    equiv_classes[ans] = conf
    return comps


def build_equiv_class_llm(question, answers, confidences=None, model='gpt-3.5-turbo', total_max_comp=12):
    start_time = time.time()
    if confidences is None:
        confidences = [1./len(answers)]*len(answers)
    equiv_classes = {}
    # print("answers, confidences", answers, confidences)
    
    comps = 0
    for i, (ans, conf) in enumerate(zip(answers, confidences)):
        # if not unique list, conf is just ans_count 
        # print("e, a, c", equiv_classes, ans, conf)
        comp = add_to_equiv_class_llm(equiv_classes, ans, conf, question, model)
        comps += comp 
        if comps >= total_max_comp:
            break 
    
    # assert that sum of the values in equiv class = 1 :)
    print(f"build_equiv_class_llm took {time.time() - start_time:.3f} seconds and {comps} comparisons")
    return equiv_classes
