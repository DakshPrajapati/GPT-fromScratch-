import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from tokenizer import GptTokenizer
from transformer.transformer import GPTLanguageModel

def load_tokenizer():
    
    tokenizer = GptTokenizer()
    tokenizer.load("./tokenizer/models/gpt.model")
    
    return tokenizer

def load_model(device='cpu'):
    
    model = GPTLanguageModel(n_embd,
                            n_head,
                            dropout,
                            block_size,
                            vocab_size,
                            n_layer
    )
    return model.to(device)

if __name__ == "__main__":
    # hyperparameters
    batch_size = 64 # how many independent sequences will we process in parallel?
    block_size = 256 # what is the maximum context length for predictions?
    max_iters = 100
    eval_interval = 50
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 64
    n_head = 6
    n_layer = 6
    dropout = 0.2
    checkpoint_steps = 500
    vocab_size = 1000

    tokenizer = load_tokenizer()
    model = load_model(device)
    text = "Hii my self bhautik pithadiya"
    encoding = [tokenizer.encode(s) for s in text.split()]
    print(encoding)
    print([tokenizer.decode(encode) for encode in encoding])
    # vocabb = GptTokenizer()
    # vocabb = vocabb.load('./tokenizer/models/gpt.vocab')
    # print(vocabb)