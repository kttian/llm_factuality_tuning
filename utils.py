from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch 


def get_precision(precision): # convert str to torch dtype
    if precision == "float16":
        dtype = torch.float16
    elif precision == "float32":
        dtype = torch.float32
    elif precision == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"precision {precision} not recognized")
    return dtype


def load_model_tok(args):
    '''
    loads model and tokenizer based on args.model_name
    args:
        args: argparse Namespace object
    '''
    model_dtype = get_precision(args.model_dtype)
    print(f"Loading model {args.model_name} with type {model_dtype}")
    if args.model_name == "llama7b" or args.model_name == "llama-7b":
        hf_model = 'huggyllama/llama-7b'
    elif args.model_name == "llama65b" or args.model_name == "llama-65b":
        hf_model = 'huggyllama/llama-65b'
    elif args.model_name == "llama30b" or args.model_name == "llama-30b":
        hf_model = 'huggyllama/llama-30b' 
    elif args.model_name == "llama2-7b":
        hf_model = 'meta-llama/Llama-2-7b-hf'
        if model_dtype != torch.bfloat16:
            print("WARNING: llama2 supports bfloat16")
    elif args.model_name == "llama2-70b":
        hf_model = 'meta-llama/Llama-2-70b-hf'
        if model_dtype != torch.bfloat16:
            print("WARNING: llama2 supports bfloat16")
    elif args.model_name == "llama2-7b-chat":
        hf_model = 'meta-llama/Llama-2-7b-chat-hf'
        if model_dtype != torch.bfloat16:
            print("WARNING: llama2 supports bfloat16")
    elif args.model_name == "pythia28":
        hf_model = 'EleutherAI/pythia-2.8b'
    elif args.model_name == "pythia69":
        hf_model = 'EleutherAI/pythia-6.9b'
    elif args.model_name == "pythia120":
        hf_model = 'EleutherAI/pythia-12b'
    else:
        raise Exception(f"model name {args.model_name} not recognized")


    quantization_config = BitsAndBytesConfig(load_in_8bit=(args.num_bits == 8),
                                             load_in_4bit=(args.num_bits == 4))
    
    if args.num_gpus == 1 and args.num_bits < 16:
        print("one gpu")
        model = AutoModelForCausalLM.from_pretrained(hf_model, cache_dir=args.cache_dir, torch_dtype=model_dtype,
                                                     quantization_config=quantization_config, use_auth_token=args.auth_token,
                                                     low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_model, cache_dir=args.cache_dir, torch_dtype=model_dtype,
                                                  quantization_config=quantization_config, use_auth_token=args.auth_token,
                                                  low_cpu_mem_usage=True)
        if args.num_bits > 8:
            model.cuda()
        print("Loading to device:", model.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_model, cache_dir=args.cache_dir, torch_dtype=model_dtype,
                                                     quantization_config=quantization_config, use_auth_token=args.auth_token,
                                                     device_map="balanced", low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_model, cache_dir=args.cache_dir, torch_dtype=model_dtype,
                                                  quantization_config=quantization_config, use_auth_token=args.auth_token,
                                                  device_map="balanced", low_cpu_mem_usage=True)
        print("Loading to device:", model.device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id 
    tokenizer.padding_side = "left"
    return model, tokenizer
