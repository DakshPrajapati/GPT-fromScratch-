import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from transformer.transformer import GPTLanguageModel
from tokenizer.gpt import GptTokenizer
from Dataset import CustomDataset

print('****** Loading Dataset *********')
DATA = pd.read_csv('./data/dataset_v2.csv')
TRAIN_DATA,VAL_DATA = train_test_split(DATA,test_size=0.3,shuffle=True, random_state=42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(n_embd = 64,
               n_head = 6,
               dropout = 6,
               block_size = 256,
               vocab_size = 10000,
               n_layer = 6):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = GPTLanguageModel(n_embd,
                             n_head,
                             dropout,
                             block_size,
                             vocab_size,
                             n_layer
    )
    
    model = model.to(device)
    
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = GptTokenizer()
    tokenizer.load(tokenizer_path)
    
    return tokenizer


@torch.no_grad()
def estimate_loss(model,eval_iters,input_batch,target_batch):
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        
        for k in tqdm(range(eval_iters)):
            X,y = input_batch,target_batch
            
            logits , loss = model(X,y)
            del X,y
            losses[k] = loss.item()
    
        out[split] = losses.mean()
    model.train()
    
    return out

if __name__ == "__main__":
    batch_size = 64 # how many independent sequences will we process in parallel?
    block_size = 256 # what is the maximum context length for predictions?
    max_iters = 100
    eval_interval = 50
    learning_rate = 3e-4
    eval_iters = 200
    n_embd = 64
    n_head = 6
    n_layer = 6
    dropout = 0.2
    checkpoint_steps = 500
    vocab_size = 10000
    
    T_path = '/home/ksuser/Bhautik/GPT/tokenizer/models/gpt.model'
    
    print('******* Loading Model *********')
    model = load_model(n_embd,
                       block_size,
                       dropout,
                       n_head,
                       n_layer,
                       vocab_size)
    model = model.to(device)    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print('******** Loading Tokenizer **********')
    tokenizer = load_tokenizer(T_path)
    
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        input_batch = pad_sequence(inputs, batch_first=True, padding_value=10001)
        target_batch = pad_sequence(targets, batch_first=True, padding_value=10001)
        return input_batch, target_batch

    train_dataset = CustomDataset(TRAIN_DATA, tokenizer, max_length=block_size)
    val_dataset = CustomDataset(VAL_DATA, tokenizer, block_size)
    
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size, shuffle=True,collate_fn=collate_fn)    
    
    
    
    # ITERS = len(tokenize_data) // batch_size
    EPOCHS = 10
    iter = 0
    for epoch in EPOCHS:
        iter = 0 if epoch==0 else iter
        for input_batch,target_batch in train_dataloader:
            
            if iter% eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(model,eval_iters,input_batch,target_batch)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
            logits,loss  = model(input_batch,target_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    
    # for epoch in tqdm(range(EPOCHS)):
    #     for iter in range(ITERS):
            
    #         if iter % eval_interval == 0 or iter == max_iters - 1:
    #             losses = estimate_loss(model,eval_iters=eval_iters)
    #             print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    #         # sample a batch of data
    #         xb, yb = get_batch('train')

    #         # evaluate the loss
    #         logits, loss = model(xb, yb)
    #         optimizer.zero_grad(set_to_none=True)
    #         loss.backward()
    #         optimizer.step()
            
    #         if (iter+1) % checkpoint_steps == 0:
    #             with open(f'./checkpoints/chkpt_{iter+1}.pkl','wb') as f:
    #                 pickle.dump(model,f)
    #             print('Checkpoints Saved')
            
    
    