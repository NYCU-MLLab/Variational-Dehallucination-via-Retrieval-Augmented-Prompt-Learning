import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
import random
import yaml
from torch.nn.utils.rnn import pad_sequence

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)




class QADataset(Dataset):
    def __init__(self,mode='train'):
        self.dataset = load_dataset(config['dataset_path'])
        self.dataset=self.dataset[mode]
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question=self.dataset[idx]['ambiguous_question'].replace('？','').replace('?','').strip()
        answer=self.dataset[idx]['annotations'][0]['long_answer'].strip()
        
        return question,answer,[answer]
    
def collect_fn(batch):

    q_a_list=[]
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
    for q,a ,_ in batch:
        example=q+'? '+a
        q_a_list.append(example)
   
    output=tokenizer(text=q_a_list,return_tensors="pt",padding=True,truncation=True,max_length=config['max_len'])
    
    return output,q_a_list

class collateLM():
    def __init__(self, max_len=config['max_len'], tokenizer=None):
        assert tokenizer is not None

        self.tokenizer = tokenizer
        self.max_len=max_len
        self.max_p_len=int(max_len*0.7)
        self.max_c_len=max_len-self.max_p_len
        
        self.bos_id=self.tokenizer.bos_token_id
        self.eos_id=self.tokenizer.eos_token_id
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.pad_id =self.tokenizer.pad_token_id

    def __call__(self, batch):
        
        tokens=[]
        masks=[]
        targets=[]
        querys=[]
        answers=[]
        all_as=[]
        q_a=[]
        for q, a ,all_a in batch:
            out=self.tokenizer(q+'? '+a+self.tokenizer.eos_token, return_tensors="pt", truncation=True, max_length=self.max_len)
            q_out=self.tokenizer(q+'?', return_tensors="pt", truncation=True, max_length=self.max_len)
            
            ids = out['input_ids'][0][1:]
            mask = out['attention_mask'][0][1:]
            target = ids.clone()
            target[:q_out['input_ids'].shape[1]-1] = -100
           
            tokens.append(ids)
            masks.append(mask)
            targets.append(target)
            querys.append(q)
            answers.append(a)
            all_as.append(all_a)
            q_a.append(q+'? '+a)
            
        tokens=pad_sequence(tokens, batch_first=True, padding_value=self.pad_id)
        masks=pad_sequence(masks, batch_first=True)
        targets=pad_sequence(targets, batch_first=True, padding_value=-100)
        return tokens, masks, targets, querys, answers,all_as,q_a


if __name__=='__main__':
    pass