import torch
from tqdm import tqdm
from torch.utils.data import random_split
import torch.nn.functional as F

from dataset import QADataset,collect_fn,collateLM
from model import Softprompt
from llm import LLaMa

import yaml
import numpy as np
from bert_score import score
from rouge_score import rouge_scorer
from metric import compute_f1
from transformers.utils import logging
logging.set_verbosity_error() 
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device=='cuda'

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)

seed = config['seed']
generator = torch.Generator()
generator.manual_seed(seed)



def trainer(epoch):
    loss_mv=4.8

    acc=[]
    token_acc=[]
   
    soft_prompt.train()
    bar = tqdm(train_loader, ncols=0, smoothing=0.05, desc=f'epoch:{epoch+1:03d}/{train_config["max_epoch"]}')
    for tokens, masks, targets, querys, answers,_ ,q_a in bar:
        
        tokens, masks, targets = map(lambda x:x.to(device),[tokens, masks, targets])

        optimizer.zero_grad()
        
     
        soft_emb=soft_prompt(tokens)

        soft_emb.retain_grad()
       
        out, loss ,target= LLM.forward(ids=tokens, target=targets, masks=masks, soft_examples=soft_emb,)



        mask_indices =target != -100
        
        for b in range(out.shape[0]):
            a=torch.masked_select(torch.argmax(out[b,:-1,:],dim=-1),mask_indices[b,1:])
            tar=torch.masked_select(target[b],mask_indices[b])
           
            token_accc= (a==tar).float().mean().cpu().numpy()
            if np.isnan(token_accc):
                token_accc=0
            token_acc.append(token_accc)
            accc=int(token_accc)
            acc.append(accc)
  
        loss.backward()
        optimizer.step()
        # print(soft_emb.grad)
        
        loss_mv = loss_mv*0.98+loss.item()*0.02
    
        bar.set_postfix_str(f'loss:{loss_mv:.3f}, acc:{sum(acc)/len(acc):.4f}, tokenacc:{sum(token_acc)/len(token_acc):.4f} ')
    scheduler.step()


@torch.inference_mode()
def inference(mode,loader,posterior=False,output_files=False):
    if output_files:
        fh = open(f'output/output_{mode}.txt', 'w')

        bar = tqdm(loader, ncols=0, smoothing=0.05, desc=mode,file=fh)
    else:
        bar = tqdm(loader, ncols=0, smoothing=0.05, desc=mode)
    acc_ls=[]
    bestsore=[]
    f1_ls=[]
    rouge_ls=[]
    rouge1_ls=[]
    rouge2_ls=[]
    soft_prompt.eval()
    es_ls=[]
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scorer1 = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    scorer2 = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)

    with torch.no_grad():
        for tokens, _, _, querys, answers,all_as,q_a in bar:
            
            
            soft_emb=soft_prompt(tokens)
            
            
            for i in range(len(querys)):
                predict,es=LLM.generate(querys[i]+'?', soft_emb[i][None,:,:], 512,streamer=False,return_eigenscore=True)
                predict=predict.strip()
                if not torch.isinf(torch.tensor(es)) and not torch.isnan(torch.tensor(es)):
                    es_ls.append(es)
                
                    
                rouge=max((scorer.score(predict, answer)['rougeL'].fmeasure) for answer in all_as[i])
                rouge1=max((scorer1.score(predict, answer)['rouge1'].fmeasure) for answer in all_as[i])
                rouge2=max((scorer2.score(predict, answer)['rouge2'].fmeasure) for answer in all_as[i])
                
                rouge_ls.append(rouge)
                rouge1_ls.append(rouge1)
                rouge2_ls.append(rouge2)
                bs = max((score([predict], [answer],lang='en', model_type='bert-base-uncased')[2]) for answer in all_as[i])
                bestsore.append(bs.item())
            
            bar.set_postfix_str(f' rouge1: {sum(rouge1_ls)/len(rouge1_ls):.4f}rouge2: {sum(rouge2_ls)/len(rouge2_ls):.4f}rougel: {sum(rouge_ls)/len(rouge_ls):.4f} bertscore: {sum(bestsore)/len(bestsore):.4f}es:{sum(es_ls)/len(es_ls):.4f}')

if __name__=='__main__':
        
    train_config=config['train_config']
    loader_config=config['dataloader_config']
    loader_config_infer=config['dataloader_config']

    model_path="meta-llama/Llama-2-7b-chat-hf"
    LLM=LLaMa(model_path)
    
    soft_prompt=Softprompt()
    soft_prompt.to(device)
    
    

    if train_config['load_E']:
        soft_prompt.load_state_dict(torch.load('save/soft_prompt.pt'))

        

    collate_fn = collateLM(tokenizer=LLM.tokenizer)
    qadataset = QADataset()
    train_size = int(train_config['spilt'] * len(qadataset))
    test_size = len(qadataset) - train_size

    train_dataset, test_dataset = random_split(qadataset, [train_size, test_size],generator=generator)
    val_dataset= QADataset(mode='dev')
    optimizer = torch.optim.AdamW(list(soft_prompt.parameters()), lr=train_config['lr'], betas=train_config['betas'], weight_decay=train_config['weight_decay'])
    
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config['max_epoch'], eta_min=train_config['eta_min'])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_config, collate_fn=collate_fn)
    val_loader=torch.utils.data.DataLoader(val_dataset, **loader_config_infer, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, **loader_config_infer, collate_fn=collate_fn)

    

    for i in range(train_config['max_epoch']):
        trainer(i)
        torch.save(soft_prompt.state_dict(),f'save/soft_prompt.pt')

        # inference(mode='validation',loader=val_loader) 
    inference(mode='test',loader=test_loader)   
    