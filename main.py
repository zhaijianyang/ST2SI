import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import logging
import torch
from torch.utils.data import DataLoader, ConcatDataset
from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin
from accelerate.utils import set_seed
import torch.distributed as dist
from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

from parameters import get_args
from utils import now_time, get_logger, create_optimizer_and_scheduler
from data_utils import load_data, tokenize_items
from data import RecDataset, ItemDataset, Collator
from models_utils import get_model_config_tokenizer, print_trainable_parameters
from trainer import Trainer


def main():
    
    args = get_args()
    set_seed(args.seed)

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # kwargs_handlers = [ddp_kwargs]
    kwargs_handlers = None
    
    deepspeed_plugin = None
    if args.deepspeed:
        hf_deepspeed_config = HfTrainerDeepSpeedConfig(args.deepspeed)
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=hf_deepspeed_config)
        
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=kwargs_handlers,\
                            deepspeed_plugin=deepspeed_plugin)
    
    logger = get_logger(args.log_file, name=__name__)
    arg_dict = vars(args)
    sorted_args = sorted(arg_dict.items(), key=lambda x: x[0])
    for arg, value in sorted_args:
        space = ' ' * (30-len(arg))
        logger.info(f'{arg}: {space} {value}')

    train, val, test, item_meta_dict, item2id = load_data(args)
    args.item_num = len(item2id)
    args.logger = logger
    args.device = accelerator.device
    
    model, config, tokenizer = get_model_config_tokenizer(args)
    
    if args.fix_backbone:
        for _ in model.decoder.parameters():
            _.requires_grad = False
            
    if args.fix_emb:
        model.decoder.embed_tokens.weight.requires_grad = False
    
    logger.info(model)
    print_trainable_parameters(args, model)
    
    if accelerator.is_local_main_process:
        tokenizer.save_pretrained(args.output_path)
        config.save_pretrained(args.output_path)

    tokenized_items = tokenize_items(item_meta_dict, tokenizer, args, accelerator)

    train_dataset = RecDataset(train, val, test, tokenized_items, args, 'train', tokenizer)
    val_dataset = RecDataset(train, val, test, tokenized_items, args, 'valid', tokenizer)
    test_dataset = RecDataset(train, val, test, tokenized_items, args, 'test', tokenizer)
    
    item_dataset = ItemDataset(tokenized_items, args, tokenizer)
    collator = Collator(args, tokenizer)
    
    train_datasets = [train_dataset]
    if args.index_alignment:
        train_datasets.append(item_dataset)
        
    train_data = ConcatDataset(train_datasets)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collator)
    
    for i in range(1):
        train_data = train_dataset.__getitem__(i)
        logger.info('[Train Data]')
        logger.info('[Sequence]: \n' + tokenizer.decode(train_data[0]))
        logger.info('[Target]: \n' + str(train_data[1]))

        val_data = val_dataset.__getitem__(i)
        logger.info('[Valid Data]')
        logger.info('[Sequence]: \n' + tokenizer.decode(val_data[0]))
        logger.info('[Target]: \n' + str(val_data[1]))

        test_data = test_dataset.__getitem__(i)
        logger.info('[Test Data]')
        logger.info('[Sequence]: \n' + tokenizer.decode(test_data[0]))
        logger.info('[Target]: \n' + str(test_data[1]))
            
    trainer = Trainer(args, accelerator, model, train_loader, dev_loader, test_loader)
    
    # logger.info(now_time() + 'First test.')
    # test_metrics = trainer.predict(trainer.test_loader)
    # logger.info(now_time() + f'==Test set==: {test_metrics}')
    
    if args.deepspeed:
        ## 一直打印信息
        import deepspeed
        deepspeed.utils.logging.logger.setLevel(logging.ERROR)  # 或者你想要的级别，如 logging.ERROR
    
    if not args.only_test:

        trainer.train()
            
    logger.info(now_time() + 'Finish.')
      
if __name__ == "__main__":
    main()