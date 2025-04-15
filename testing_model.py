from tokenizer.gpt import RegexTokenizer
from transformer.main import GPTLanguageModel,Block,MultiHeadAttention
import torch,pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## loading Model
# model = GPTLanguageModel()
# model = model.to(device)

checkpoint_path = './checkpoints/chkpt_25001.pkl'
with open(checkpoint_path, 'rb') as f:
    model = pickle.load(f)

model.to(device)
print("Model loaded successfully from", checkpoint_path)

## Loading Tokenizer
path = './tokenizer/models/nolan/gpt.model'
tokenizer = RegexTokenizer()
tokenizer.load(path)

prompt = 'Who is Christopher'
input_tokens = tokenizer.encode(prompt)

context = torch.tensor(input_tokens,dtype=torch.long,device = device)
output = model.generate(context.unsqueeze(0),max_new_tokens =70)[0].tolist()
print(output[0])
print(tokenizer.decode(output))