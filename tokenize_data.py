import concurrent.futures
import pandas as pd
from tqdm import tqdm
import ultraimport
import torch
import json

# Load and set up the tokenizer
from tokenizer.gpt import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.load('./tokenizer/models/nolan/gpt.model')
special_tokens = {'<eos>': 10000, '<pad>': 10001}
tokenizer.register_special_tokens(special_tokens)

# Define your tokenization function
def tokenize_data(data):
    # print(data.split(' '))
    inputs = data.split(' ')
    input_tokens = inputs[:-1]
    input_ids = tokenizer.encode(" ".join(tokens for tokens in input_tokens))
    
    target_tokens = inputs[1:]
    target_ids = tokenizer.encode(' '.join(tokens for tokens in target_tokens))
    # print("Before",len(input_ids),len(target_ids))
    if len(input_ids) > len(target_ids):
        target_ids+= [special_tokens['<pad>']] * (len(input_ids) - len(target_ids))
    elif len(target_ids) > len(input_ids):
        input_ids+= [special_tokens['<pad>']] * (len(target_ids) - len(input_ids))
    # print(len(input_ids),len(target_ids))
    return [input_ids,target_ids]

# Read your dataset
DATA = pd.read_csv('./data/dataset_v2.csv')  # Your dataset here
DATA = DATA
# Add an index column to keep track of the original order
DATA['index'] = DATA.index

# Define the number of processes
num_processes = 47

# Create a new DataFrame to store tokenized data
tokenized_data = pd.DataFrame(columns=["X", "y", "index"])
tqdm.pandas()
# Define a function for parallel tokenization
def tokenize_chunk(chunk):
    tokenized_chunk = pd.DataFrame(columns=["X", "y", "index"])
    # temp = pd.DataFrame(columns=["combined", "index"])
    # print(chunk['Text'].apply(tokenize_data)[0])
    # tokenized_chunk['X'],tokenized_chunk['y'] = chunk['Text'].apply(tokenize_data)
    tokenized_results = chunk['Text'].progress_apply(tokenize_data).apply(pd.Series)
    tokenized_chunk['X'] = tokenized_results[0]
    tokenized_chunk['y'] = tokenized_results[1]
    
    tokenized_chunk['index'] = chunk['index']
    # print(tokenize_chunk['X'][0],tokenize_chunk['Y'][0])
    # print(temp)
    return tokenized_chunk

# Split the dataset into chunks for parallel processing
chunk_size = len(DATA) // num_processes
chunks = [DATA.iloc[i:i+chunk_size] for i in range(0, len(DATA), chunk_size)]

# Process each chunk in parallel using ProcessPoolExecutor
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(tokenize_chunk, chunk) for chunk in chunks]

    # Gather results from all futures
    tokenized_chunks = [future.result() for future in concurrent.futures.as_completed(futures)]

# Concatenate results from all chunks
# print(type(tokenize_chunk))
tokenized_data = pd.concat(tokenized_chunks)

# Sort by the original index to maintain the order
tokenized_data = tokenized_data.sort_values(by='index').reset_index(drop=True)

# Drop the index column as it is no longer needed
tokenized_data = tokenized_data.drop(columns=['index'])
tokenized_data['X'] = tokenized_data['X'].apply(json.dumps)
tokenized_data['y'] = tokenized_data['y'].apply(json.dumps)

# Save the tokenized data
print('Saving Data')
tokenized_data.to_csv('./data/tokenized_data_v3.csv', index=False,sep ='|')
