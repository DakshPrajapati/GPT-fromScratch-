import wikipediaapi
from tqdm.auto import tqdm
from collections import Counter
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import pandas as pd

wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (xyz@gmail.com)', 'en', timeout=1000)

train_pages = ['Christopher_Nolan',
                'Following',
                'Memento',
                'Insomnia',
                'Batman Begins',
                'The Prestige',
                'The Dark Knight',
                'Inception',
                'The Dark Knight Rises',
                'Interstellar',
                'Dunkirk',
                'Tenet',
                'Oppenheimer']

def get_wiki_sections_text(page):
    ignore_sections = ["References", "See also", "External links", "Further reading", "Sources"]
    wiki_page = wiki_wiki.page(page)
    
    # Get all the sections text
    page_sections = [x.text for x in wiki_page.sections if x.title not in ignore_sections and x.text != ""]
    section_titles = [x.title for x in wiki_page.sections if x.title not in ignore_sections and x.text != ""]
    
    # Add the summary page
    page_sections.append(wiki_page.summary)
    section_titles.append("Summary")

    return page_sections, section_titles

def get_pages_df(pages):
    page_section_texts = []
    for page in tqdm(pages):
        sections, titles = get_wiki_sections_text(page)
        for section, title in zip(sections, titles):
            page_section_texts.append({
                'page': page,
                'section_title': title,
                'text': section
            })
    print(len(page_section_texts))
    return pd.DataFrame(page_section_texts)

def creating_sliding_windows(df):
    columns = ['page','section_title','text']
    dataset = pd.DataFrame(columns=columns)
    length_of_sliding_window = 256
    page = df.page
    section_title = df.section_title
    text = df.text
    text = text.split()
    
    total_number_of_windows = len(text)//length_of_sliding_window
    
    start = 0
    end = length_of_sliding_window
    
    for i in range(total_number_of_windows+1):
        if i==total_number_of_windows:
            sliding_window = text[start:]
        else:
            sliding_window = text[start:end]
        row = {
            'page':[page],
            'section_title' : [section_title],
            'text':[' '.join(s for s in sliding_window)]
        }
        start+=length_of_sliding_window
        end +=length_of_sliding_window
        
        data = pd.DataFrame(row)
        
        dataset = pd.concat([dataset,data],ignore_index=True)
    
    return dataset  

if __name__ == '__main__':
    
    train_pages_df = get_pages_df(train_pages)
    
    columns = ['page','section_title','text']
    dataset = pd.DataFrame(columns=columns)
    
    for i in range(len(train_pages_df)):
        dataset = pd.concat([dataset,creating_sliding_windows(train_pages_df.iloc[i])])
    
    dataset.reset_index(drop=True, inplace=True)
    
    dataset.to_csv('train_pages.csv',index=False)