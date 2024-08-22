import torch
import torch.nn as nn
from transformers import AutoModel ,AutoTokenizer
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from torch.utils.data import random_split
from bert_score import score
from dataset import QADataset,collect_fn,collateLM
from model import PromptEncoder
from retriever import Retriever
from llm import LLaMa
import yaml
import torch.nn.functional as F

from bert_score import score
from rouge_score import rouge_scorer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device=='cuda'

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)

seed = config['seed']
generator = torch.Generator()
generator.manual_seed(seed)

def compute_f1(prediction, truth):
    pred_tokens = prediction
    truth_tokens = truth
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    f1=2 * (prec * rec) / (prec + rec)
    return f1

def trainer(epoch):
    loss_mv=4.8
    acc=[]
    token_acc=[]
    prior_PE.train()
    
    bar = tqdm(train_loader, ncols=0, smoothing=0.05, desc=f'epoch:{epoch+1:03d}/{train_config["max_epoch"]}')
    i=0
    for tokens, masks, targets, querys, answers,_ ,q_a in bar:
        i+=1
        tokens, masks, targets = map(lambda x:x.to(device),[tokens, masks, targets])

        optimizer.zero_grad()
        '''
        retrive topk examples
        '''
        topk_examples_prior=retriever.retrieve(querys)# list Bxn

        '''
        examples to token to embs
        '''
        topk_token_prior=LLM.tokenizer(topk_examples_prior, return_tensors="pt",padding=True,add_special_tokens=False)
      
       
        topk_embs_prior=LLM.model.model.embed_tokens(topk_token_prior.input_ids.to(device))# Bxnx4096
        
        
        '''
        use topk embs to get soft examples
        '''
        prior_mean =prior_PE(topk_embs_prior.to(torch.float),topk_token_prior.attention_mask.to(torch.float).to(device))
        
        prior_mean.retain_grad()
        
        '''
        feed soft examples and q to LLM
        '''
        
        out, loss ,target= LLM.forward(ids=tokens, target=targets, masks=masks, soft_examples=prior_mean)
        
        mask_indices =target != -100
        
        for b in range(out.shape[0]):
            a=torch.masked_select(torch.argmax(out[b,:-1,:],dim=-1),mask_indices[b,1:])
            tar=torch.masked_select(target[b],mask_indices[b])
           
            token_accc= (a==tar).float().mean()
            
            token_acc.append(token_accc)
            accc=int(token_accc)
            acc.append(accc)
  
        loss.backward()
        optimizer.step()

        loss_mv = loss_mv*0.98+loss.item()*0.02
        
        bar.set_postfix_str(f'loss:{loss_mv:.3f}, acc:{sum(acc)/len(acc):.4f}, tokenacc:{sum(token_acc)/len(token_acc):.4f} ')
        
    scheduler.step()


@torch.inference_mode()
def inference(mode,loader):
    
    bar = tqdm(loader, ncols=0, smoothing=0.05, desc=mode)
    acc_ls=[]
    bestsore=[]
    f1_ls=[]
    rouge_ls=[]
    es_ls=[]
    prior_PE.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    with torch.no_grad():
        for _, _, _, querys, answers,all_as,_ in bar:
            
            topk_examples=retriever.retrieve(querys,train=False)
            topk_token=LLM.tokenizer(topk_examples, return_tensors="pt",padding=True,add_special_tokens=False)
            
            topk_embs=LLM.model.model.embed_tokens(topk_token.input_ids.to(device))# Bxnx4096

            prior_mean=prior_PE(topk_embs.to(torch.float),topk_token.attention_mask.to(torch.float).to(device))
            
            for i in range(len(querys)):
                predict,es=LLM.generate(querys[i]+'?', prior_mean[i][None,:,:], 8,streamer=False,return_eigenscore=True)
                predict=predict.strip()
                found = predict in all_as[i]
                if not torch.isinf(torch.tensor(es)) and not torch.isnan(torch.tensor(es)):
                    es_ls.append(es)
               
                if found:
                    acc=1
                else: 
                    acc=0

                acc_ls.append(acc)
                f1_score = max((compute_f1(predict, answer)) for answer in all_as[i])
                rouge=max((scorer.score(predict, answer)['rougeL'].fmeasure) for answer in all_as[i])
                f1_ls.append(f1_score)
                rouge_ls.append(rouge)
                bs = max((score([predict], [answer],lang='en', model_type='bert-base-uncased')[2]) for answer in all_as[i])
                bestsore.append(bs.item())
            
            bar.set_postfix_str(f'EM: {sum(acc_ls)/len(acc_ls):.4f} f1: {sum(f1_ls)/len(f1_ls):.4f} rouge: {sum(rouge_ls)/len(rouge_ls):.4f} bertscore: {sum(bestsore)/len(bestsore):.4f} es:{sum(es_ls)/len(es_ls):.4f}')
            
        
if __name__=='__main__':
        
    train_config=config['train_config']
    loader_config=config['dataloader_config']
   
    retriever = Retriever(k=train_config['topk'])
    model_path="meta-llama/Llama-2-7b-chat-hf"
    LLM=LLaMa(model_path)
    
    prior_PE=PromptEncoder()
    prior_PE.to(device)

    if train_config['load_E']:
        prior_PE.load_state_dict(torch.load('save/prior_PE.pt'))

    collate_fn = collateLM(tokenizer=LLM.tokenizer)
    qadataset = QADataset()
    train_size = int(train_config['spilt'] * len(qadataset))
    test_size = len(qadataset) - train_size

    train_dataset, test_dataset = random_split(qadataset, [train_size, test_size],generator=generator)
    val_dataset=QADataset(mode='validation')
    optimizer = torch.optim.AdamW(prior_PE.parameters(), lr=train_config['lr'], betas=train_config['betas'], weight_decay=train_config['weight_decay'])
    
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config['max_epoch'], eta_min=train_config['eta_min'])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_config, collate_fn=collate_fn)
    val_loader=torch.utils.data.DataLoader(val_dataset, **loader_config, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, **loader_config, collate_fn=collate_fn)


    for i in range(train_config['max_epoch']):
        trainer(i)
        torch.save(prior_PE.state_dict(),'save/prior_PE.pt')
        # inference(mode='validation',loader=val_loader)
    inference(mode='test',loader=test_loader)
