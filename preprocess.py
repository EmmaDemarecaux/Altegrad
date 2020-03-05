"""
# Preprocessing functions to be used for the different methods afterwards 
"""
import os
import codecs
from bs4 import BeautifulSoup
import re
from nltk.stem.snowball import FrenchStemmer
import collections


def clean_host_texts(data, tok, stpwds, punct, verbosity=5):
    cleaned_data = []
    for counter, host_text in enumerate(data):
        # converting article to lowercase already done
        temp = BeautifulSoup(host_text, 'lxml')
        text = temp.get_text()  # removing HTML formatting
        text = text.replace('{html}', '')
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = text.translate(str.maketrans(punct, ' '*len(punct)))
        text = ''.join([l for l in text if not l.isdigit()])
        text = re.sub(' +', ' ', text)  # striping extra white space
        text = text.strip()  # striping leading and trailing white space
        tokens = tok.tokenize(text)  # tokenizing (splitting based on whitespace)
        tokens = [token for token in tokens if (token not in stpwds) and (len(token) > 1)]
        stemmer = FrenchStemmer()   
        tokens_stemmed = []
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed
        cleaned_data.append(tokens)
        if counter % round(len(data)/verbosity) == 0:
            print(counter, '/', len(data), 'text cleaned')
    return [' '.join(l for l in sub_cleaned_data) for 
            sub_cleaned_data in cleaned_data]


def remove_duplicates(train_file='train.csv'):
    with open(train_file, 'r') as f:
        train_data_ids = f.read().splitlines()

    train_data_ids = list(set(train_data_ids))
    duplicates_to_drop = [item for item, count in
                          collections.Counter([item.split(",")[0] for item
                                               in train_data_ids]).items() if count > 1]
    # Remove duplicates in training data: hosts + labels
    train_hosts = list()
    y_train = list()
    for row in train_data_ids:
        host, label = row.split(",")
        if host not in duplicates_to_drop:
            train_hosts.append(host)
            y_train.append(label.lower())
    return train_hosts, y_train 


def import_texts(texts_path):
    texts = dict()
    file_names = os.listdir(texts_path)
    for filename in file_names:
        with codecs.open(os.path.join(texts_path, filename), encoding='utf8', errors='ignore') as f:
            texts[filename] = f.read().replace('\n', '').lower()
    return texts
            
            
def generate_data(data_hosts, texts):
    """Get textual content of web hosts of the training set"""
    data = list()
    for host in data_hosts:
        if host in texts:
            data.append(texts[host])
        else:
            data.append('')  
    return data
