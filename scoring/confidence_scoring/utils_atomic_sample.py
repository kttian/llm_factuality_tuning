from scipy.stats import entropy

import sys
sys.path.append('../../')
from parsing import group_ans_simple, normalize_ans

def is_equiv(question, text1, text2):
    return normalize_ans(text1) == normalize_ans(text2)

def compute_se_scores_from_samples(samples=None, last_unit=None, reward_mode=None):
    # if reward mode is given, return that reward. otherwise, return dict of all rewards
    # currently only using exact match
    if samples is None:
        return {}
    
    samples_norm = [normalize_ans(s) for s in samples]
    samples_norm, probs = group_ans_simple(samples_norm)

    reward_dict = {}
    ent = entropy(probs)
    reward_dict['entropy'] = ent
    
    if last_unit:
        if normalize_ans(last_unit) in samples_norm:
            lu_idx = samples_norm.index(last_unit)
            lu_prob = probs[lu_idx]
        else:
            lu_prob = 0
        reward_dict['confidence_sampled'] = lu_prob

    max_c = max(probs)
    reward_dict['confidence_max'] = max_c
    return reward_dict if reward_mode is None else reward_dict[reward_mode]

def get_pronouns():
    file = "/iris/u/kattian/project_hallucination/sft_rl_0719/se_rewards/names_pronouns_400.txt"
    with open(file, 'r') as f:
        names_pronouns = f.read().split("\n")
    pronouns = [x.split(",")[1].strip() for x in names_pronouns]
    return pronouns

def get_aq_sampling_few_shot_examples_bio():
    prompt = ""
    prompt += "What college did Hillary Clinton attend? Wellesley College\n"
    prompt += "Who did Hillary Clinton marry? Bill Clinton\n"
    prompt += "What political position does Hillary Clinton hold? Secretary of State\n"
    prompt += "What kind of background was Hillary Clinton raised in? upper middle class\n"

    prompt += "What is LeBron James' profession? basketball player\n"
    prompt += "Where does LeBron James rank among NBA players? one of the best\n"
    prompt += "In what city was LeBron James born? Akron\n"
    prompt += "Which team did LeBron James begin his NBA career with? Cleveland Cavaliers\n"

    prompt += "Where was Yo-Yo Ma born? Paris\n"
    prompt += "When was Yo-Yo Ma born? 1955\n"
    prompt += "What profession were Yo-Yo Ma's parents? musicians\n"
    return prompt

def get_aq_sampling_few_shot_examples_medqa():
    prompt = ""
    prompt += "What is the most common symptom of frostbite? Pain\n"
    prompt += "At what age does menopause typically occur? 50\n"
    prompt += "Do the symptoms of frostbite include waxy skin? Yes\n"
    prompt += "Down syndrome is caused by what? Genetic mutation\n"
    prompt += "What do the symptoms of Down syndrome include? Intellectual disability, low muscle tone, and small hands\n"
    prompt += "Where is edema most commonly seen? Feet and legs\n"
    prompt += "At what age can the symptoms of narcolepsy begin? Any age\n"
    return prompt 

def get_aq_sampling_prompt(question, dataset_name, context_qs=None, context_as=None):
    '''
    We sample answers from the base model (e.g., Llama) to each atomic question. This function 
    constructs the few shot example for that. 
    '''
    if context_qs is None:
        context_qs = []
    if context_as is None:
        context_as = []
    
    if dataset_name == "bio":
        prompt = get_aq_sampling_few_shot_examples_bio()
    elif dataset_name == "medqa" or dataset_name == "toy":
        prompt = get_aq_sampling_few_shot_examples_medqa()
    else:
        raise ValueError("Invalid dataset name")

    for i in range(len(context_qs)):
        context_q = context_qs[i]
        context_a = context_as[i]
        prompt += f"{context_q} {context_a}\n"
    
    prompt += f"{question}" # don't use trailing space!
    return prompt

if __name__ == "__main__":
    aq_sampling_prompt = get_aq_sampling_prompt("EXAMPLE QUESTION", "toy")
    print("AQ sampling prompt:", aq_sampling_prompt)
