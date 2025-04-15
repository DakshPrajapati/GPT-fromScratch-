import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from Dataset import CustomDataset
from tokenizer.gpt import RegexTokenizer
import torch.multiprocessing as mp

import pickle
from tqdm import tqdm
import pandas as pd


tokenizer_path = './tokenizer/models/nolan/gpt.model'
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 512 # what is the maximum context length for predictions?
# max_iters = 100
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 250
n_embd = 64
n_head = 6
n_layer = 6
dropout = 0.2 
checkpoint_steps = 5000
vocab_size = 10002


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

print('Loading Dataset')
DATA = pd.read_csv('./data/tokenized_data_v3.csv',sep='|')
# DATA = DATA[:1000]
DATA['X'] = DATA['X'].apply(json.loads)
DATA['y'] = DATA['y'].apply(json.loads)

# DATA = DATA[:64]
print('Loading Tokenzier')
tokenizer = RegexTokenizer()
tokenizer.load(tokenizer_path)
special_tokens = {
    '<eos>' : 10000,
    '<pad>': 10001
}
tokenizer.register_special_tokens(special_tokens)



print('Train Test Split')
TRAIN_DATA,VAL_DATA = train_test_split(DATA,test_size=0.3,shuffle=True, random_state=42)

print('Custom Dataset')
train_dataset = CustomDataset(TRAIN_DATA, tokenizer, max_length=block_size)
val_dataset = CustomDataset(VAL_DATA, tokenizer, block_size)

def collate_fn(batch):
    # Separate the input and target sequences
    input_ids, target_ids = zip(*batch)
    
    # Pad the sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=special_tokens['<pad>'])
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=special_tokens['<pad>'])
    
    return input_ids_padded, target_ids_padded

print('DataLoader')
torch.set_num_threads(10)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size, shuffle=True,collate_fn=collate_fn)    

TOTAL_ITERATION = len(train_dataloader)
del DATA,TRAIN_DATA,VAL_DATA
eval_iters = 2000

@torch.no_grad()
def estimate_loss(model,eval_iters):
    counter = 0
    out = {}
    model.eval()
    
    
    losses = torch.zeros(eval_iters)
    
    for split in ['train','val']:
        if split == 'val':
            k = 0
            
            for input_batch, target_batch in tqdm(val_dataloader, total=eval_iters, desc=f"Estimating loss for val"):
                if k==eval_iters:
                    break
                logits , loss = model(input_batch,target_batch)
                losses[k] = loss.item()
                k+=1
                

            out[split] = losses.mean()
        else:
            k = 0
            for input_batch, target_batch in tqdm(train_dataloader, total=eval_iters, desc=f"Estimating loss for train"):
                if k==eval_iters:
                    break
                logits , loss = model(input_batch,target_batch)
                losses[k] = loss.item()
                k+=1

            out[split] = losses.mean()
    model.train()
    
    return out


def train_epoch(epoch,iter):
    losses = {}
    iter = 0 if epoch==0 else iter
    # print(train_dataloader.is_cuda)
    for input_batch,target_batch in tqdm(train_dataloader, total=TOTAL_ITERATION, desc=f"Training {epoch+1}"):
        # print(f'Batch {iter+1}')
        
    
        logits,loss  = model(input_batch,target_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()      
        iter+=1
        # losses['train'] = train_loss.mean()
        if iter % (TOTAL_ITERATION/2) == 0 :
            losses = estimate_loss(model,eval_iters)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
        if (iter) % checkpoint_steps == 0:
            with open(f'./checkpoints/chkpt_{iter+1}.pkl','wb') as f:
                pickle.dump(model,f)
            print('Checkpoints Saved')


def train(model):
    EPOCHS = 200
    iter = 0

    for epoch in range(EPOCHS):
        # print('In the training loop')
        # print(F'********** EPOCH {epoch} ***********')
        train_epoch(epoch,iter)
        
        

if __name__ == "__main__":
    print("Loading Model")
    model = GPTLanguageModel()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train(model)
    
    # num_process = 4
    # print('Model Sharing')
    # # model.share_memory()
    # print('Model sharing complete')
    # EPOCHS = 10
    # Multi Processing
    # processes = []
    # for rank in range(num_process):
    #     print('In the loop')
    #     p = mp.Process(target=train,args=(model,))
    #     p.start()
    #     processes.append(p)
    
    # for p in processes:
    #     p.join()
    # train(model)
    