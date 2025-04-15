import pandas as pd
import concurrent.futures
from tqdm import tqdm


def load_data():
    raw_data = open('./data/dataset_v1.txt')
    raw_data = raw_data.readlines()
    
    raw_data_split = [data.split(' ') for data in raw_data]
    
    raw_data_joined = " ".join([y for x in raw_data_split for y in x])
    
    raw_data_words = raw_data_joined.split(' ')
    
    return raw_data_words

def process_data(x, block_size):
    data_df = pd.DataFrame(columns=["X", "y"])  # Initialize an empty DataFrame
    i = 0
    while i < len(x) - block_size:
        X = " ".join(x[i:i+block_size])
        y = " ".join(x[i+1: i+block_size+1])
        data = {
            "X": X,
            "y": y
        }
        # print(data)
        data_df = pd.concat([data_df, pd.DataFrame([data])])
        i += 1
    return data_df

def main():
    # Your list of words
    print('************* Loading Data ******************')
    x = load_data()  # Your list of words here

    # Define block_size and number of processes
    block_size = 256
    num_processes = 45

    # Split the list into chunks for parallel processing
    print('********** Splitting into chunks **************')
    chunk_size = len(x) // num_processes
    chunks = [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)]

    # Process each chunk in parallel using ProcessPoolExecutor
    print('************ Parallel Processing ***************')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        with tqdm(total=len(chunks)) as pbar:
            for chunk in chunks:
                futures.append(executor.submit(process_data, chunk, block_size))
                pbar.update(1)

        # Gather results from all futures
        data_dfs = []
        for future in concurrent.futures.as_completed(futures):
            data_dfs.append(future.result())
        data_dfs = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Concatenate results from all chunks
    final_data_df = pd.concat(data_dfs)

    # Now you have your final DataFrame with processed data
    final_data_df.to_csv('./data/dataset_v1.csv',index=False)

if __name__ == "__main__":
    main()
