import json 
import warnings

example_names_medqa = ["menopause", "breast cancer", "varicose veins"]
example_statements_medqa = [
    [
        "Menopause is a time in a woman's life.",
        "Menopause is the time when a woman no longer has menstrual periods.",
        "There is a decline in the ovarian hormone estrogen.",
    ],
    [
        "The signs and symptoms include a lump or thickening in or near the breast or underarm.",
        "The signs and symptoms include a change in the size or shape of the breast.",
    ],
    [
        "Varicose veins occur when the veins under the skin become enlarged.",
        "Veins in the legs lose their ability to efficiently circulate blood back to the heart."
        "Varicose veins often appear as bulging.",
    ]
]
example_questions_medqa = [
    [
        "Menopause is a time in whose life?",
        "Menopause is the time when a woman no longer has what?",
        "During menopause there is a decline in what?"
    ],
    [
        "Do the signs and symptoms of breast cancer include a lump or thickening in or near the breast or underarm?",
        "Do the signs and symptoms of breast cancer include a change in the size or shape of the breast?"
    ],
    [
        "Varicose veins occur when what happens to the veins under the skin?",
        "Varicose veins in the legs lose their ability to do what?",
        "What do varicose veins often appear as?",
    ]
]
example_answers_medqa = [
    ["A woman's", "menstrual periods", "estrogen"],
    ["yes", "yes"],
    ["they become enlarged", "efficiently circulate blood back to the heart", "bulging"]
]

example_names_bio = ["Hillary Clinton", "LeBron James"]
example_statements_bio = [
    [
        "Hillary Clinton was born in 1947.",
        "Hillary attended the Wellesley College.",
        "She married Bill Clinton."
    ],
    [
        "LeBron James is a professional basketball player.",
        "He is one of the best in the NBA.",
        "James was born in Akron."
    ]
]
example_questions_bio = [
    [
        "In what year was Hillary Clinton born?",
        "What college did Hillary Clinton attend?",
        "Who did Hillary Clinton marry?"
    ],
    [
        "What is LeBron James' profession?",
        "Where does LeBron James rank among NBA players?",
        "In what city was LeBron James born?"
    ]
]
example_answers_bio = [
    ["1947", "Wellesley College", "Bill Clinton"],
    ["basketball player", "one of the best", "Akron"]
]

# get instruction for just atomic question, for max confidence scoring
def get_instruction_medqa(name):
    # TODO: add code for handling the capitalization of name 
    prompt = f"I will provide a statement containing one atomic fact about the medical condition {name}."
    prompt += f" Please rephrase the following statement into a specific question testing knowledge of the key fact in the statement."
    prompt += " For example:\n\n"
    return prompt 

# get instruction for both atomic question AND answer, version sampled confidence scoring 
def get_instruction_medqa_SC(name):
    prompt = f"I will provide a statement containing one atomic fact about the medical condition {name}."
    prompt += f" Please rephrase the following statement into a question and short answer pair that tests for knowledge of the key fact in the statement."
    prompt += " The answer should be a single word or short phrase."
    return prompt


# get instruction for just atomic question, for max confidence scoring
def get_instruction_bio(name, pronoun="them"):
    # TODO: add code for picking the correct pronoun
    prompt = f"I will provide a statement containing one atomic fact related to {name} or people around {pronoun}."
    prompt += f" Please rephrase the following statement into a specific question testing knowledge of the key fact in the statement."
    prompt += " For example:\n\n"
    return prompt

# get instruction for both atomic question AND answer, version for sampled confidence scoring 
def get_instruction_bio_SC(name, pronoun="them"):
    prompt = f"I will provide a statement containing one atomic fact related to {name} or people around {pronoun}."
    prompt += f" Please rephrase the following statement into a question and answer pair that tests for knowledge of the key fact in the statement."
    prompt += " The answer should be a single word or short phrase."
    return prompt 

# biography dataset prompt with no few shot examples
def get_prompt_aq_bio_SC_v0(name, pronoun, new_statement):
    inst = get_instruction_bio_SC(name, pronoun)
    inst += f"\nStatement: {new_statement}"
    return inst


# medical dataset prompt with few shot examples
def get_prompt_aq_med_SC(name, new_statement):
    inst = f"I will provide the name of a medical condition and statement containing one atomic fact related to that medical condition."
    inst += f" Please rephrase the following statement into a question and answer pair that tests for knowledge of the key fact in the statement."
    inst += " The answer should be a single word or short phrase."
    inst += " For example:\n"
    inst += f"\nMedical Condition: {example_names_medqa[0]}"
    inst += f"\nStatement: {example_statements_medqa[0][2]}"
    inst += f"\nQuestion: {example_questions_medqa[0][2]}"
    inst += f"\nAnswer: {example_answers_medqa[0][2]}"
    inst += "\n"
    inst += f"\nMedical Condition: {example_names_medqa[1]}"
    inst += f"\nStatement: {example_statements_medqa[1][1]}"
    inst += f"\nQuestion: {example_questions_medqa[1][1]}"
    inst += f"\nAnswer: {example_answers_medqa[1][1]}"
    inst += "\n"
    inst += f"\nMedical Condition: {name}"
    inst += f"\nStatement: {new_statement}"
    return inst

# biography dataset prompt with few shot examples 
def get_prompt_aq_bio_SC(name, new_statement):
    inst = f"I will provide a person's name and statement containing one atomic fact related to the person or people around them."
    inst += f" Please rephrase the following statement into a question and answer pair that tests for knowledge of the key fact in the statement."
    inst += " The answer should be a single word or short phrase."
    inst += " For example:\n"
    inst += f"\nName: {example_names_bio[0]}"
    inst += f"\nStatement: {example_statements_bio[0][2]}"
    inst += f"\nQuestion: {example_questions_bio[0][2]}"
    inst += f"\nAnswer: {example_answers_bio[0][2]}"
    inst += "\n"
    inst += f"\nName: {example_names_bio[1]}"
    inst += f"\nStatement: {example_statements_bio[1][1]}"
    inst += f"\nQuestion: {example_questions_bio[1][1]}"
    inst += f"\nAnswer: {example_answers_bio[1][1]}"
    inst += "\n"
    inst += f"\nName: {name}"
    inst += f"\nStatement: {new_statement}"
    return inst


# atomic question AND answer, version for sampled confidence scoring
def get_convert_to_aqa_prompt(name, new_statement, dataset, topic_info):
    if dataset == "medqa":
        name = topic_info[name]["lower_case"]
        return get_prompt_aq_med_SC(name, new_statement)
    elif dataset == "bio":
        return get_prompt_aq_bio_SC(name, new_statement)
    else:
        raise ValueError(f"DATASET_NAME must be 'medqa' or 'bio', but got {dataset}")

# just atomic question, for max confidence scoring
def get_convert_to_aq_prompt(name, new_statement, DATASET_NAME, topic_info,
                             context_statements=None, context_questions=None,
                             ):
    '''
    This function builds the prompt that we use to ask GPT-3.5 to convert a statement to a question.

    Prompt Format:
    I will provide a statement containing one atomic fact about the medical condition menopause. 
    Please rephrase the following statement into a specific question testing knowledge of the key 
    fact in the statement. For example:

    Statement: Menopause is a time in a woman's life.
    Question: Menopause is a time in whose life?

    Statement: Menopause is the time when a woman no longer has menstrual periods.
    Question: Menopause is the time when a woman no longer has what?

    Statement: There is a decline in the ovarian hormone estrogen.
    Question: During menopause there is a decline in what?

    [Repeat the above for 2 example names]

    I will provide a statement containing one atomic fact about the medical condition [NEW TOPIC]. 
    Please rephrase the following statement into a specific question testing knowledge of the key 
    fact in the statement. For example:

    Statement: Context S1. # sometimes earlier atomic facts are helpful context for the new atomic fact
    Question: Context Q1? 

    Statement: Context S2.
    Question: Context Q2?

    Statement: [NEW STATEMENT (ATOMIC FACT)]
    Question:

    '''
    # handle default case for context statements
    if context_statements is None or context_questions is None:
        context_statements = []
        context_questions = []
    
    # get the correct example names, statements, questions, and instruction based on the dataset
    if DATASET_NAME == "medqa" or DATASET_NAME == "toy":
        example_names = example_names_medqa
        example_statements = example_statements_medqa
        example_questions = example_questions_medqa
        def get_instruction(name):
            if name not in topic_info:
                warnings.warn(f"name '{name}' not found in topic_info")
                name_lower = name 
            else:
                name_lower = topic_info[name]["lower_case"]
            return get_instruction_medqa(name_lower)
    elif DATASET_NAME == "bio":
        example_names = example_names_bio
        example_statements = example_statements_bio
        example_questions = example_questions_bio
        def get_instruction(name):
            if name not in topic_info:
                warnings.warn(f"name '{name}' not found in topic_info")
                pronoun = "them"
            else:
                pronoun = topic_info[name]["pronoun"]
            return get_instruction_bio(name, pronoun)
    else:
        raise ValueError(f"DATASET_NAME must be 'medqa' or 'bio', but got {DATASET_NAME}")
    assert (
        len(example_names) == len(example_statements) == len(example_questions)
    ), "Lengths of example names, statements, and questions must be the same."

    # build few shot prompt from the example statements and questions given for each example name/topic
    prompt = ""
    for i in range(len(example_names)):
        prompt += get_instruction(example_names[i])
        for j in range(len(example_statements[i])):
            prompt += f"Statement: {example_statements[i][j]}\n"
            prompt += f"Question: {example_questions[i][j]}\n\n"

    prompt += get_instruction(name)
    
    for i in range(len(context_statements)):
        prompt += f"Statement: {context_statements[i]}\n"
        prompt += f"Question: {context_questions[i]}\n\n"
    
    prompt += f"Statement: {new_statement}\n"
    prompt += f"Question:"
    return prompt

if __name__ == "__main__":
    topic_info_file = "/iris/u/kattian/project_hallucination/factual-rl/dataset_toy/topic_info.json"
    with open(topic_info_file, 'r') as f:
        topic_info = json.load(f)
        print(f"Loaded topic_info file {topic_info_file} with {len(topic_info)} topics.")
    
    print(">>Example for MEDQA:")
    inst = get_instruction_medqa("<EXAMPLE CONDITION>")
    print("instruction:")
    print(inst)

    print("Do it again to test warn once:")
    inst = get_instruction_medqa("<EXAMPLE CONDITION>")
    print("instruction:")
    print(inst)

    prompt = get_convert_to_aq_prompt("<NEW CONDITION>", "<NEW STATEMENT>", "medqa", topic_info)
    print("prompt:")
    print(prompt)
    
    print("\n" + "="*50 + "\n")
    print(">>Example for BIOS:")
    inst = get_instruction_bio("<EXAMPLE CONDITION>")
    print("instruction:")
    print(inst)

    prompt = get_convert_to_aq_prompt("<NEW CONDITION>", "<NEW STATEMENT>", "bio", topic_info)
    print("prompt:")
    print(prompt)
