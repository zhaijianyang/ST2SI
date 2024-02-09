from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, LongformerModel
import os
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import bitsandbytes as bnb

from utils import now_time
from models import OPTRecModel

def get_model_config_tokenizer(args):
    
    if args.checkpoint_dir is not None:
        kwargs = {}
        args.model_name_or_path = args.checkpoint_dir
    else:
        kwargs = {"cache_dir": args.model_cache_dir, "local_files_only": os.path.exists(args.model_cache_dir)}
    
    ######## ======== tokenizer ==========
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **kwargs)
    if 'llama' in args.model_name_or_path:
        tokenizer.pad_token_id = 0
        
    if args.query_token_num == 0:
        args.query_token_ids = []
    else:
        if args.query_token_mode == 0 or args.query_token_mode == 1:
            special_tokens = [f'<query_0_{j}>' for j in range(args.query_token_num)]
            if args.checkpoint_dir is None:
                tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            query_token_ids = [tokenizer.convert_tokens_to_ids(f'<query_0_{j}>') for j in range(args.query_token_num)]
            
        elif args.query_token_mode == 2:
            special_tokens = [f'<query_{i}_{j}>' for i in range(args.max_item_num) for j in range(args.query_token_num)]
            if args.checkpoint_dir is None:
                tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            query_token_ids = []
            for i in range(args.max_item_num):
                query_token_ids.append([tokenizer.convert_tokens_to_ids(f'<query_{i}_{j}>') for j in range(args.query_token_num)])
                
            args.logger.info(now_time() + f'add special token num: {args.max_item_num}')
            
        args.query_token_ids = query_token_ids
    
    alignment_ids = []
    if args.index_alignment:
        if args.query_same:
            alignment_ids = query_token_ids
        else:
            special_tokens = [f'<alignment_{i}>' for i in range(args.query_token_num)]
            if args.checkpoint_dir is None:
                tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            alignment_ids = [tokenizer.convert_tokens_to_ids(token) for token in special_tokens]
        
    args.alignment_ids = alignment_ids
        
    for key, value in tokenizer.special_tokens_map.items():
        args.logger.info(f'{key}: {value}; {tokenizer.convert_tokens_to_ids(value)}')
    
    ######## ======== config ==========
    config = AutoConfig.from_pretrained(args.model_name_or_path, **kwargs)
    config.linear_dim = args.item_num
    
    ######## ======== model ==========
    if args.checkpoint_dir is None:
        model = OPTRecModel.from_pretrained(args.model_name_or_path, config=config, **kwargs)
        model.resize_token_embeddings(len(tokenizer))
    else:
        args.logger.info(now_time() + f'Load from {args.checkpoint_dir}.')
        model = OPTRecModel(config)
        model.set_linear(config.linear_dim)
        if args.load_model is not None:
            state_dict = torch.load(os.path.join(args.checkpoint_dir, args.load_model + '.bin'), map_location='cpu')
        else:
            state_dict = torch.load(args.checkpoint_dir + '/pytorch_model.bin', map_location='cpu')
        state_dict.pop('score.weight')
        info = model.load_state_dict(state_dict, strict=False)
        args.logger.info(now_time() + f'Load info: {info}')
    
    config.vocab_size = len(tokenizer)
    
    if args.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        
    if args.use_lora:

        if args.target_modules == None:
            args.target_modules = find_all_linear_names(args, model)

        args.logger.info(now_time() + f'Adding LoRA modules: {args.target_modules}')
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.bias,
            # task_type='CAUSAL_LM' if 'opt' in args.model_name_or_path else "SEQ_2_SEQ_LM",
        )
        model = get_peft_model(model, config)

    # print_trainable_parameters(args, model)
    
    return model, config, tokenizer

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
        
    if 'score' in lora_module_names:
        lora_module_names.remove('score')
        
    return list(lora_module_names)

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            # args.logger.info(_, param.numel())
            trainable_params += param.numel()
    if args.bits == 4: 
        trainable_params /= 2

    args.logger.info(now_time() + 
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )