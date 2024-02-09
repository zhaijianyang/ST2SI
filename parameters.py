import os
from argparse import ArgumentParser
    
def get_args():
    parser = ArgumentParser()
    # debug
    parser.add_argument('--not_debug', action='store_true')
    # data and log path
    parser.add_argument('--checkpoint_dir', type=str, default=None, required=False)
    parser.add_argument('--load_model', type=str, default=None, required=False)
    parser.add_argument('--data_root', type=str, default='/datasets/datasets/RecFormer/finetune_data', required=False)
    parser.add_argument('--dataset', type=str, default='Scientific', required=False)
    parser.add_argument('--output_dir', type=str, default='./log')
    parser.add_argument('--suffix', type=str, default='debug')

    # data process
    parser.add_argument('--train_attr', nargs='+', default=['title'], help=['title', 'brand', 'category', 'image'])
    parser.add_argument('--max_attr_length', type=int, default=32)
    parser.add_argument('--max_item_num', type=int, default=20)
    parser.add_argument('--max_token_num', type=int, default=512)
    
    ## pretrain
    parser.add_argument('--pretrain_datasets', nargs='+', default=[], 
                        help=['Automotive', 'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry', 'Electronics', 
                              'Grocery_and_Gourmet_Food', 'Home_and_Kitchen', 'Movies_and_TV'])  ## 'CDs_and_Vinyl' for validate
    parser.add_argument('--fix_backbone', action='store_true')
    parser.add_argument('--fix_emb', action='store_true')

    # model
    parser.add_argument('--model_name_or_path', type=str, default='facebook/opt-125m', help='')
    parser.add_argument('--model_cache_dir', type=str, default='/datasets/pretrain_models')
    
    # tokenizer
    parser.add_argument('--pad_right', action='store_true')
    
    # virtual token
    parser.add_argument('--query_token_mode', type=int, default=1, help={0: 'only end pos', 1: 'every pos same', 2: 'every pos diff'})
    parser.add_argument('--query_token_num', type=int, default=1, help=[1]) # 
    parser.add_argument('--index_alignment', action='store_true')
    parser.add_argument('--query_same', action='store_true') # 
    parser.add_argument('--no_prompt', action='store_true') # 

    # train
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deepspeed', type=str, default=None)
    parser.add_argument('--mixed_precision', type=str, default='fp16')
    parser.add_argument('--num_train_epochs', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_checkpointing_enable', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=5e-5)
    parser.add_argument('--base_weight_decay', type=float, default=0)
    parser.add_argument('--linear_lr', type=float, default=5e-5)
    parser.add_argument('--linear_weight_decay', type=float, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--skip_valid', type=int, default=0)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--patient', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=100)
    
    ## valid and test
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10], help='ks for Metric@k')
    parser.add_argument('--valid_metric', type=str, default='NDCG@10')
    parser.add_argument('--only_test', action='store_true')

    # lora
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--target_modules', nargs='+', default=None, help=["q_proj", "v_proj"])
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--bias', type=str, default="none")
    parser.add_argument('--bits', type=int, default=16, help=[4, 8, 16, 32])
    
    args = parser.parse_args()
    
    model_prefix = args.model_name_or_path.replace("/", "-")
    args.output_path = os.path.join(args.output_dir, args.dataset, model_prefix + '_' + args.suffix)
    if args.dataset == 'pretrain':
        datas = '_'.join(args.pretrain_datasets)
        args.output_path = os.path.join(args.output_dir, args.dataset, datas, model_prefix + '_' + args.suffix)
        
    if args.checkpoint_dir is not None:
        pretrain_dataset = args.checkpoint_dir.split('/')[-2]
        args.output_path = os.path.join(args.output_dir, 'finetune', pretrain_dataset, args.dataset, model_prefix + '_' + args.suffix)
        
        if args.load_model is not None:
            args.output_path = args.output_path + args.load_model

    if args.only_test:
        if not os.path.exists(args.checkpoint_dir):
            raise 'checkpoint_dir not exit ...'
        args.output_path = args.checkpoint_dir
        args.log_file = os.path.join(args.output_path, 'test_' + args.dataset + '.log')
    else:
        args.log_file = os.path.join(args.output_path, 'train.log')
    
    os.makedirs(args.output_path, exist_ok=True)
    
    return args