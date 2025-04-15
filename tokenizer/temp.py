import os
import pandas as pd

dataset_path = '../NOLAN_wiki/imdb/preprocessed_dataset/'
filenames = os.listdir(dataset_path)
print(filenames)

# Wiki dataset
text = open("./wiki_dataset.txt", "r", encoding="utf-8").read()

# IMDB Reviews

for name in filenames:
    df = pd.read_csv(dataset_path+name)
    for x in df['Review']:
        text += x
    print(f'Appended {name}')

print(text)
with open('nolan_dataset','w+') as f:
    f.write(text)