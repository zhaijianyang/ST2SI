from torch.utils.data import Dataset
import random
import torch
import copy

class RecDataset(Dataset):
    def __init__(self, interactions, tokenized_items, args, mode, tokenizer=None):

        '''
        user2train: dict of sequence data, user--> item sequence
        '''
        
        self.interactions = interactions
        self.tokenized_items = tokenized_items
        self.max_item_num = args.max_item_num
        self.max_token_num = args.max_token_num
        self.mode = mode
        self.tokenizer = tokenizer
        self.args = args
        self.query_token_ids = args.query_token_ids

        self.users = [user for user, seq in self.interactions.items() if len(seq) > 1]
        
        if not args.not_debug:
            self.users = self.users[:16]
            self.args.logger.info(f'[Debug: {mode}], user num: {len(self.users)}')
        else:
            self.args.logger.info(f'[Mode: {mode}], user num: {len(self.users)}')

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):

        user = self.users[index]
        
        if self.mode == 'train':
            items_seq_all = self.interactions[user][:-1]
            start_pos = random.randint(0, len(items_seq_all) - 2)
            target_pos = random.randint(start_pos + 1, len(items_seq_all) - 1)

            items_seq = items_seq_all[start_pos:target_pos]
            label = items_seq_all[target_pos]

        else:
            items_seq = self.interactions[user][:-1]
            label = self.interactions[user][-1]

        items_seq = items_seq[::-1]  # reverse items order
        items_seq = items_seq[:self.max_item_num]

        items_tokens_list = []
        new_item_seq = []

        count = 0
        for item_id in items_seq:
            item_tokens = self.tokenized_items[item_id]['item_tokens']

            count += len(item_tokens)
            if count < self.max_token_num:
                items_tokens_list.append(item_tokens)
                new_item_seq.append(item_id)

            else:
                break
            
        items_tokens_list = items_tokens_list[::-1] # reverse items order
        new_item_seq = new_item_seq[::-1]
        
        target_items = new_item_seq[1:] + [label]

        input_ids = []
        target_ids = []
            
        for idx, item_tokens in enumerate(items_tokens_list):

            input_ids.extend(item_tokens)
            target_ids.extend([-100] * len(item_tokens))
            
            if self.args.query_token_mode != 0:
                if self.args.query_token_num == 0:
                    target_ids[-1] = target_items[idx]
                    
                else:
                    if self.args.query_token_mode == 1:
                        input_ids.extend(self.query_token_ids)
                        temp = (len(self.query_token_ids) - 1) * [-100] + [target_items[idx]]
                        target_ids.extend(temp)
                        
                    elif self.args.query_token_mode == 2:
                        input_ids.extend(self.query_token_ids[idx])
                        temp = (len(self.query_token_ids[idx]) - 1) * [-100] + [target_items[idx]]
                        target_ids.extend(temp)
                
        if self.args.query_token_mode == 0:
            if self.args.query_token_num == 0:
                target_ids[-1] = target_items[-1]
            else:
                input_ids.extend(self.query_token_ids)
                temp = (len(self.query_token_ids) - 1) * [-100] + [target_items[-1]]
                target_ids.extend(temp)
            
        assert len(input_ids) == len(target_ids)

        return input_ids, target_ids, label
    
class ItemDataset(Dataset):
    def __init__(self, tokenized_items, args, tokenizer):

        self.tokenized_items = tokenized_items
        self.args = args
        self.alignment_ids = args.alignment_ids
        
        prompt = "Please give the index of the following item: "
        self.prompt_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))

    def __len__(self):
        return len(self.tokenized_items)

    def __getitem__(self, index):

        tokenized_item = self.tokenized_items[index]['item_tokens']

        input_ids = []
        target_ids = []
        label = index
                
        input_ids.extend(tokenized_item)
        input_ids.extend(self.alignment_ids)
        
        target_ids.extend([-100] * len(tokenized_item))
        
        if self.args.query_token_num == 0:
            target_ids[-1] = label
        else:
            target_ids.extend([-100] * (len(self.alignment_ids) - 1) + [label])
            
        input_ids = self.prompt_ids + input_ids
        target_ids = [-100] * len(self.prompt_ids) + target_ids
            
        assert len(input_ids) == len(target_ids)

        return input_ids, target_ids, label
    
class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def __call__(self, batch):

        batch_num = len(batch)
        max_1 = max([len(sample[0]) for sample in batch])
        
        input_seq_ids = torch.ones(batch_num, max_1, dtype=torch.long) * self.tokenizer.pad_token_id
        target_seq_ids = torch.ones(batch_num, max_1, dtype=torch.long) * -100
        labels = []
        
        for idx, sample in enumerate(batch):
            num = len(sample[0])
            if not self.args.pad_right:
                input_seq_ids[idx, -num:] = torch.LongTensor(sample[0])
                target_seq_ids[idx, -num:] = torch.LongTensor(sample[1])

            else:
                input_seq_ids[idx, :num] = torch.LongTensor(sample[0])
                target_seq_ids[idx, :num] = torch.LongTensor(sample[1])
                
            labels.append(sample[2])
                
        seq_attention_mask = input_seq_ids != self.tokenizer.pad_token_id
        
        labels = torch.LongTensor(labels)
        
        return input_seq_ids, seq_attention_mask, target_seq_ids, labels