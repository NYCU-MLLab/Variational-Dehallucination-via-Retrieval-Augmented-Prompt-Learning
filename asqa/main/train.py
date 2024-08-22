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
from torch.nn.utils.rnn import pad_sequence
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device=='cuda'

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)

seed = config['seed']
generator = torch.Generator()
generator.manual_seed(seed)

def kl_loss(mean1, logvar1, mean2, logvar2):
        
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

def kl_anneal_function(itera, epoch,num_epochs, start_beta, end_beta,dataloader_len):
    anneal_rate = (end_beta - start_beta) / (num_epochs*dataloader_len)
    current_beta = start_beta + anneal_rate * (itera*1+epoch*dataloader_len)
    
    if epoch>3:
        current_beta=1
    return current_beta


def trainer(epoch):
    loss_mv=4.8
    acc=[]
    token_acc=[]
    prior_PE.train()
    posterior_PE.train()
    
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

        topk_examples_posterior=retriever.retrieve(q_a)
        '''
        examples to token to embs
        '''
        
        topk_tokens=LLM.tokenizer(topk_examples_prior+topk_examples_posterior, return_tensors="pt",padding=True,add_special_tokens=False)
        
        batchsize=int(topk_tokens.input_ids.shape[0]/2)
        
        topk_embs_prior=LLM.model.model.embed_tokens(topk_tokens.input_ids.to(device)[:batchsize])# Bxnx4096
        
        topk_embs_posterior=LLM.model.model.embed_tokens(topk_tokens.input_ids.to(device)[batchsize:])


        
        '''
        use topk embs to get soft examples
        '''
        prior_mean, prior_logvar=prior_PE(topk_embs_prior.to(torch.float),topk_tokens.attention_mask.to(torch.float).to(device)[:batchsize])
        prior_logvar=torch.log(torch.tensor(1e-6))
        
        posterior_mean, posterior_logvar=posterior_PE(topk_embs_posterior.to(torch.float),topk_tokens.attention_mask.to(torch.float).to(device)[batchsize:])
        posterior_logvar=torch.log(torch.tensor(1e-6))
        z=posterior_PE.reparameter(posterior_mean, posterior_logvar)
        
        z.retain_grad()
        kl = nn.MSELoss()(posterior_mean,prior_mean)
        
        '''
        feed soft examples and q to LLM
        '''
        
        out, loss ,target= LLM.forward(ids=tokens, target=targets, masks=masks, soft_examples=z)
        loss+=1*kl
        
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


        loss_mv = loss_mv*0.98+loss.item()*0.02
        
        bar.set_postfix_str(f'loss:{loss_mv:.3f}, acc:{sum(acc)/len(acc):.4f}, tokenacc:{sum(token_acc)/len(token_acc):.4f}, kl:{kl:.2e}')
    scheduler.step()


@torch.inference_mode()
def inference(mode,loader):
    
    bar = tqdm(loader, ncols=0, smoothing=0.05, desc=mode)
    acc=[]
    prior_PE.eval()
    
    with torch.no_grad():
        for _, _, _, querys, answers,all_as,_ in bar:
            
            topk_examples=retriever.retrieve(querys,train=False,)
            
            
            topk_token=LLM.tokenizer(topk_examples, return_tensors="pt",padding=True,add_special_tokens=False)
            
            topk_embs=LLM.model.model.embed_tokens(topk_token.input_ids.to(device))# Bxnx4096
            topk_embs=F.pad(topk_embs,[0,0,0,5],mode='constant',value=0)
            prior_mean, prior_logvar=prior_PE(topk_embs.to(torch.float))
            
            for i in range(len(querys)):
                predict=LLM.generate(querys[i]+'?', prior_mean[i][None,:,:], 8,streamer=False)
                predict=predict.strip()
                found = predict in all_as[i]
                
                if found:
                    accc=1
                else: 
                    accc=0

                acc.append(accc)
                
            bar.set_postfix_str(f'acc: {sum(acc)/len(acc):.4f} ')
        
if __name__=='__main__':
        
    train_config=config['train_config']
    loader_config=config['dataloader_config']
   
    retriever = Retriever(k=train_config['topk'])
    model_path="meta-llama/Llama-2-7b-chat-hf"
    LLM=LLaMa(model_path)
    
    prior_PE=PromptEncoder()
    posterior_PE=PromptEncoder()
    prior_PE.to(device)
    posterior_PE.to(device)


    if train_config['load_E']:
        prior_PE.load_state_dict(torch.load('save/prior_PE.pt'))
        # posterior_PE.load_state_dict(torch.load('save/posterior_PE.pt'))

    collate_fn = collateLM(tokenizer=LLM.tokenizer)
    qadataset = QADataset()
    train_size = int(train_config['spilt'] * len(qadataset))
    test_size = len(qadataset) - train_size

    train_dataset, test_dataset = random_split(qadataset, [train_size, test_size],generator=generator)
    val_dataset= QADataset(mode='dev')
    optimizer = torch.optim.AdamW(list(prior_PE.parameters())+list(posterior_PE.parameters()), lr=train_config['lr'], betas=train_config['betas'], weight_decay=train_config['weight_decay'])
    
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config['max_epoch'], eta_min=train_config['eta_min'])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_config, collate_fn=collate_fn)
    val_loader=torch.utils.data.DataLoader(val_dataset, **loader_config, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, **loader_config, collate_fn=collate_fn)


    for i in range(train_config['max_epoch']):
        trainer(i)
        torch.save(prior_PE.state_dict(),f'save/prior_PE.pt')
    # torch.save(posterior_PE.state_dict(),f'save/posterior_PE.pt')
        # inference(mode='validation',loader=val_loader) 
    # inference(mode='test',loader=test_loader)