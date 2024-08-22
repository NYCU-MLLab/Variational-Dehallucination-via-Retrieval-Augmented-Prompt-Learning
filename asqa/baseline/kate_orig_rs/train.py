import torch
from tqdm import tqdm

from torch.utils.data import random_split
# from bert_score import score
from dataset import QADataset,collateLM
from retriever import Retriever
from llm import LLaMa
import yaml

from bert_score import score
from rouge_score import rouge_scorer
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
def compute_f1(prediction, truth):
    pred_tokens = prediction
    truth_tokens = truth
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    f1=2 * (prec * rec) / (prec + rec)
    return f1


@torch.inference_mode()
def inference(loader,mode='orig',output_files=False):
    
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
    rouge2_ls=[]
    rouge1_ls=[]
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scorer1 = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scorer2 = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    with torch.no_grad():
        for _, _, _, querys, _,all_as,_ in bar:
            
            if mode=='kate':
                topk_examples=retriever.retrieve(querys,train=False,random_select=False,largest=True)
            if mode=='rand':
                rand_examples=retriever.retrieve(querys,train=False,random_select=True,largest=True)

            LLM.system_prompt = "Please answer the question."
            
            for i in range(len(querys)):
                if mode=='orig':
                    predict,es=LLM.generate(f'<s>[INST] <<SYS>>\n{LLM.system_prompt}\n<</SYS>>\n\n'+f'User:{querys[i].strip()}? [/INST] Assistant:',soft_examples=None,max_new_tokens= 512,streamer=False,return_eigenscore=True)
                if mode=='kate':
                    predict,es=LLM.generate(topk_examples[i]+'\n'+querys[i]+'?',soft_examples=None,max_new_tokens= 512,streamer=False,return_eigenscore=True)
                if mode=='rand':
                    predict,es=LLM.generate(rand_examples[i]+'\n'+querys[i]+'?',soft_examples=None,max_new_tokens= 512,streamer=False,return_eigenscore=True)

                if not torch.isinf(torch.tensor(es)) and not torch.isnan(torch.tensor(es)):
                    es_ls.append(es)

                if mode=='kate' or mode=='rand':
                    predict=predict.split('\n')[0].strip()
                
                rouge=max((scorer.score(predict, answer)['rougeL'].fmeasure) for answer in all_as[i])
                rouge1=max((scorer1.score(predict, answer)['rouge1'].fmeasure) for answer in all_as[i])
                rouge2=max((scorer2.score(predict, answer)['rouge2'].fmeasure) for answer in all_as[i])
                rouge_ls.append(rouge)
                rouge1_ls.append(rouge1)
                rouge2_ls.append(rouge2)
                bs = max((score([predict], [answer],lang='en', model_type='bert-base-uncased')[2]) for answer in all_as[i])
                bestsore.append(bs.item())  
            
            bar.set_postfix_str(f' rouge1: {sum(rouge1_ls)/len(rouge1_ls):.4f} rouge2: {sum(rouge2_ls)/len(rouge2_ls):.4f} rougel: {sum(rouge_ls)/len(rouge_ls):.4f} bertscore: {sum(bestsore)/len(bestsore):.4f}es: {sum(es_ls)/len(es_ls):.4f}')
            
if __name__=='__main__':
        
    train_config=config['train_config']
    loader_config=config['dataloader_config']
   
    retriever = Retriever(k=train_config['topk'])
    model_path="meta-llama/Llama-2-7b-chat-hf"
    LLM=LLaMa(model_path)
    

    collate_fn = collateLM(tokenizer=LLM.tokenizer)
    qadataset = QADataset()
    train_size = int(train_config['spilt'] * len(qadataset))
    test_size = len(qadataset) - train_size

    train_dataset, test_dataset = random_split(qadataset, [train_size, test_size],generator=generator)
    
   
    test_loader = torch.utils.data.DataLoader(test_dataset, **loader_config, collate_fn=collate_fn)

    inference(loader=test_loader,mode='orig')
    inference(loader=test_loader,mode='kate')
    inference(loader=test_loader,mode='rand')
