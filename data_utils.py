import os
import json
import torch
from multiprocessing import Pool
from tqdm import tqdm

amazon18_dataset2fullname = {
    'Beauty': 'All_Beauty',
    'Fashion': 'AMAZON_FASHION',
    'Appliances': 'Appliances',
    'Arts': 'Arts_Crafts_and_Sewing',
    'Automotive': 'Automotive',
    'Books': 'Books',
    'CDs': 'CDs_and_Vinyl',
    'Cell': 'Cell_Phones_and_Accessories',
    'Clothing': 'Clothing_Shoes_and_Jewelry',
    'Music': 'Digital_Music',
    'Electronics': 'Electronics',
    'Gift': 'Gift_Cards',
    'Food': 'Grocery_and_Gourmet_Food',
    'Home': 'Home_and_Kitchen',
    'Scientific': 'Industrial_and_Scientific',
    'Kindle': 'Kindle_Store',
    'Luxury': 'Luxury_Beauty',
    'Magazine': 'Magazine_Subscriptions',
    'Movies': 'Movies_and_TV',
    'Instruments': 'Musical_Instruments',
    'Office': 'Office_Products',
    'Garden': 'Patio_Lawn_and_Garden',
    'Pet': 'Pet_Supplies',
    'Pantry': 'Prime_Pantry',
    'Software': 'Software',
    'Sports': 'Sports_and_Outdoors',
    'Tools': 'Tools_and_Home_Improvement',
    'Toys': 'Toys_and_Games',
    'Games': 'Video_Games'
}

class LabelField:
    def __init__(self):
        self.label2id = dict()
        self.label_num = 0

    def get_id(self, label):
        
        if label in self.label2id:
            return self.label2id[label]
        
        self.label2id[label] = self.label_num
        self.label_num += 1

        return self.label2id[label]

def read_json(path, as_int=False):
    with open(path, 'r') as f:
        raw = json.load(f)
        if as_int:
            data = dict((int(key), value) for (key, value) in raw.items())
        else:
            data = dict((key, value) for (key, value) in raw.items())
        del raw
    return data

def load_pretrain_data(args):

    pretrain_datasets = [amazon18_dataset2fullname[dataset] for dataset in args.pretrain_datasets]
    args.logger.info('[preprocess pretrain data ...]')
    train = json.load(open(os.path.join(args.data_root, 'train_dict.json'), 'r'))
    train_data = {}
    for user, seq in tqdm(train.items()):
        dataname = user.split('**')[1]
        if dataname in pretrain_datasets:
            train_data[user] = seq
            
    del train
    
    item_field = LabelField()
    new_train = {}
    for user, seq in train_data.items():
        seq_ids = [item_field.get_id(item) for item in seq]
        new_train[user] = seq_ids
        
    item2id = item_field.label2id

    item_meta_dict = json.load(open(os.path.join(args.data_root, 'meta_data.json')))
    
    item_id_meta_dict = {}
    for item, item_id in item2id.items():
        item_id_meta_dict[item_id] = item_meta_dict[item]
        
    del item_meta_dict

    return new_train, item_id_meta_dict

def load_data(args):

    train = read_json(os.path.join(args.data_root, args.dataset, 'train.json'), True)
    val = read_json(os.path.join(args.data_root, args.dataset, 'val.json'), True)
    test = read_json(os.path.join(args.data_root, args.dataset, 'test.json'), True)
    item_meta_dict = json.load(open(os.path.join(args.data_root, args.dataset, 'meta_data.json')))
    
    item2id = read_json(os.path.join(args.data_root, args.dataset, 'smap.json'))

    item_meta_dict_filted = dict()
    for asin, v in item_meta_dict.items():
        if asin in item2id:
            # item_meta_dict_filted[asin] = v
            item_meta_dict_filted[item2id[asin]] = v

    return train, val, test, item_meta_dict_filted, item2id

def tokenize_items(item_meta_dict, tokenizer, args, accelerator):

    item_id2tokens = {}
    for item_id, item_attrs in tqdm(item_meta_dict.items(), ncols=100, desc='Tokenize Items', disable=(not accelerator.is_local_main_process)):
        
        item_tokens = []
            
        for idx, attr_name in enumerate(args.train_attr):

            attr_value = item_attrs[attr_name]
            gap = ': '
            
            if idx == 0:
                attr_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(attr_value))
            else:
                attr_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + attr_name + gap + attr_value))
                
            attr_tokens = attr_tokens[:args.max_attr_length]
            item_tokens += attr_tokens
            
        if args.query_token_num == 0:
            item_tokens.append(tokenizer.convert_tokens_to_ids('.'))
            
        item_id2tokens[int(item_id)] = {'item_tokens': item_tokens}
            
    return item_id2tokens
