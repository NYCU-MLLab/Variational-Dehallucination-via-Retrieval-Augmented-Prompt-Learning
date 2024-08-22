import torch
from tqdm import tqdm
from torch.utils.data import random_split
import torch.nn.functional as F

from dataset import QADataset,collect_fn,collateLM
from model import PromptEncoder
from retriever import Retriever
from llm import LLaMa
from CLUB import CLUB

import yaml

from bert_score import score
from rouge_score import rouge_scorer
from metric import compute_f1

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
    mi_ls=[]
    prior_PE.train()
    mi_estimator.train()

    bar = tqdm(train_loader, ncols=0, smoothing=0.05, desc=f'epoch:{epoch+1:03d}/{train_config["max_epoch"]}')
    
    for tokens, masks, targets, querys, answers,_ ,q_a in bar:
        
        tokens, masks, targets = map(lambda x:x.to(device),[tokens, masks, targets])

        optimizer.zero_grad()
        optimizer_mi.zero_grad()
        '''
        retrive topk examples
        '''
        topk_examples = retriever.retrieve(querys)# list Bxn
        bottomk_examples = retriever.retrieve(querys,train=False,largest=False)

        '''
        examples to token to embs
        '''
        
        topk_tokens = LLM.tokenizer(topk_examples, return_tensors="pt",padding=True,add_special_tokens=False)

        bottomk_tokens = LLM.tokenizer(bottomk_examples, return_tensors="pt",padding=True,add_special_tokens=False)

        
        topk_embs=LLM.model.model.embed_tokens(topk_tokens.input_ids.to(device))# Bxnx4096
        
        bottomk_embs=LLM.model.model.embed_tokens(bottomk_tokens.input_ids.to(device))# Bxnx4096


        '''
        use topk embs to get soft examples
        '''
        
        topk_mean, _ = prior_PE(topk_embs.to(torch.float),topk_tokens.attention_mask.to(torch.float).to(device))
        bottomk_mean, _ = prior_PE(bottomk_embs.to(torch.float),bottomk_tokens.attention_mask.to(torch.float).to(device))

        topk_mean.retain_grad()
        bottomk_mean.retain_grad()

        '''
        feed soft examples and q to LLM
        '''
        
        out, loss ,target,h_normal= LLM.forward(ids=tokens, target=targets, masks=masks, soft_examples=topk_mean,output_hidden_states=True)
        _, _,_,h_hallu= LLM.forward(ids=tokens, target=targets, masks=masks, soft_examples=bottomk_mean,output_hidden_states=True)

        
        log_like=mi_estimator.learning_loss(h_normal.detach(),h_hallu.detach())
        log_like.backward()
        optimizer_mi.step()

        
        mi_loss=mi_estimator(h_normal,h_hallu)
        beta=1e-2
        alpha=1
        loss=alpha*loss+beta*mi_loss


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
        # print(prior_PE.fc[0].weight.grad)
        
        loss_mv = loss_mv*0.98+loss.item()*0.02
        mi_ls.append(mi_loss.item())
        
        bar.set_postfix_str(f'loss:{loss_mv:.3f}, mi:{sum(mi_ls)/len(mi_ls):.4f}, acc:{sum(acc)/len(acc):.4f}, tokenacc:{sum(token_acc)/len(token_acc):.4f} ')
    torch.save(mi_ls,'save/mi_ls.pt')
    scheduler.step()
    scheduler_mi.step()

@torch.inference_mode()
def inference(mode,loader,output_files=False):
    if output_files:
        fh = open(f'output/output_{mode}.txt', 'w')

        bar = tqdm(loader, ncols=0, smoothing=0.05, desc=mode,file=fh)
    else:
        bar = tqdm(loader, ncols=0, smoothing=0.05, desc=mode)
    acc_ls=[]
    bestsore=[]
    f1_ls=[]
    rouge_ls=[]
    es_ls=[]
    prior_PE.eval()
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    with torch.no_grad():
        for _, _, _, querys, answers,all_as,q_a in bar:
            
            topk_examples=retriever.retrieve(querys,train=False,largest=True)
            topk_token=LLM.tokenizer(topk_examples, return_tensors="pt",padding=True,add_special_tokens=False)
            topk_embs=LLM.model.model.embed_tokens(topk_token.input_ids.to(device))# Bxnx4096
            
            prior_mean, _=prior_PE(topk_embs.to(torch.float))
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
        
    train_config=config['train2_config']
    loader_config=config['dataloader2_config']
    loader_config_infer=config['dataloader_config_inference']

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
    val_dataset= QADataset(mode='validation')
    optimizer = torch.optim.AdamW(list(prior_PE.parameters()), lr=train_config['lr'], betas=train_config['betas'], weight_decay=train_config['weight_decay'])
    
    
   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config['max_epoch'], eta_min=train_config['eta_min'])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_config, collate_fn=collate_fn)
    val_loader=torch.utils.data.DataLoader(val_dataset, **loader_config_infer, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, **loader_config_infer, collate_fn=collate_fn)

    mi_estimator=CLUB()
    mi_estimator=mi_estimator.bfloat16()
    mi_estimator.to(device)
    optimizer_mi = torch.optim.AdamW(mi_estimator.parameters(), lr=train_config['lr'], betas=train_config['betas'], weight_decay=train_config['weight_decay'])
    scheduler_mi = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mi, T_max=train_config['max_epoch'], eta_min=train_config['eta_min'])


    for i in range(train_config['max_epoch']):
        trainer(i)
        torch.save(prior_PE.state_dict(),f'save/prior_PE.pt')

        # inference(mode='validation',loader=val_loader) 
    inference(mode='test',loader=test_loader)    
    
