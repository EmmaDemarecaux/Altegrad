# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:13:26 2020

@author: Khaoula Belahsen
"""

# Loading the suitable librairies 
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import codecs
import string
from nltk.tokenize import TweetTokenizer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np 

# Reading the files 

# Read training data
with open("train.csv", 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
y_train = {}
for row in train_data:
    host, label = row.split(",")
    train_hosts.append(host)
    y_train[host] = label.lower()


# Load the textual content of a set of webpages for each host into the dictionary "text". 
# The encoding parameter is required since the majority of our text is french.
texts = dict()

filenames = os.listdir('text/text')
    

splittingtxt = '__________________________________________________________________'
filenameformat = '#.txt'

def newfout(inputfile, filenum):
    
    filename = 'data/' + inputfile + filenameformat.replace('#','_' + str(filenum) )
    fout = open(filename,'w')
    return fout

def split_train(filenames, splittingtxt, train_hosts):
    texts = dict()
    y_train_txt = list()  
    for filename in filenames : 
        if filename in train_hosts : 
            texts[filename] = []
            file = open(os.path.join('text/text/', filename), errors= 'ignore')
            lines = file.readlines()
            filenum=1
            fout = newfout(filename, filenum)

            for line in lines:
                if splittingtxt in line:
                    fout.close()
                    with codecs.open(os.path.join('data/', filename + filenameformat.replace('#','_' + str(filenum) )), encoding="utf8", errors='ignore') as f: texts[filename].append(f.read().replace("\n", "").lower())
                    filenum+=1
                    fout = newfout(filename, filenum)
                    
                else:
                    fout.write(line)
        
            fout.close()
            with codecs.open(os.path.join('data/', filename + filenameformat.replace('#','_' + str(filenum) )), encoding="utf8", errors='ignore') as f: texts[filename].append(f.read().replace("\n", "").lower())
            
            for i in range(filenum):
                y_train_txt.append(y_train[filename])
    return texts, y_train_txt

def split_test(filenames, splittingtxt, test_hosts):
    texts = dict()
    for filename in filenames : 
        if filename in test_hosts : 
            texts[filename] = []
            file = open(os.path.join('text/text/', filename), errors= 'ignore')
            lines = file.readlines()
            filenum=1
            fout = newfout(filename, filenum)

            for line in lines:
                if splittingtxt in line:
                    fout.close()
                    with codecs.open(os.path.join('data/', filename + filenameformat.replace('#','_' + str(filenum) )), encoding="utf8", errors='ignore') as f: texts[filename].append(f.read().replace("\n", "").lower())
                    filenum+=1
                    fout = newfout(filename, filenum)
                    
                else:
                    fout.write(line)
        
            fout.close()
            with codecs.open(os.path.join('data/', filename + filenameformat.replace('#','_' + str(filenum) )), encoding="utf8", errors='ignore') as f: texts[filename].append(f.read().replace("\n", "").lower())

    return texts

texts, y_train_txt = split_train(filenames, splittingtxt, train_hosts)

train_data_list = list()
for k, v in enumerate(texts):
    train_data_list.append(texts[v])

        
train_data_flat = []
for sublist in train_data_list:
    for item in sublist:
        train_data_flat.append(item)

def clean_host_texts(data, tokenizer, stpwds, punct, verbosity=5):
    cleaned_data = []
    for counter, host_text in enumerate(data):
        # converting article to lowercase already done
        temp = BeautifulSoup(host_text, 'lxml')
        text = temp.get_text() # removing HTML formatting
        text = ''.join(l for l in text if l not in punct) # removing punctuation
        text = ''.join([l for l in text if not l.isdigit()])
        text = re.sub(' +', ' ', text) # striping extra white space
        text = text.strip() # striping leading and trailing white space
        tokens = tokenizer.tokenize(text) # tokenizing (spliting based on whitespace)
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

# TOkenizer etc
    
tokenizer = TweetTokenizer()
punct = string.punctuation.replace('-', '’“”.»«')
stpwds = stopwords.words('french')
cleaned_train_data_txts = clean_host_texts(data=train_data_flat, tokenizer=tokenizer, 
                                     stpwds=stpwds, punct=punct)

X_train, X_eval, Y_train, Y_eval = train_test_split(
        cleaned_train_data_txts, y_train_txt, test_size=0.2, random_state=42
        )    

# Tests sets 

with open("test.csv", 'r') as f:
    test_hosts = f.read().splitlines()
    
test_texts = split_test(filenames, splittingtxt, test_hosts)

test_data_list = list()
for k, v in enumerate(test_texts):
    test_data_list.append(test_texts[v])

        
test_data_flat = []
for sublist in test_data_list:
    for item in sublist:
        test_data_flat.append(item)
        
cleaned_test_data = clean_host_texts(data=test_data_flat, tokenizer=tokenizer, 
                                     stpwds=stpwds, punct=punct)

# ========================= Choosing classifier
clf_nb = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', ngram_range=(1, 2))),
                    ('clf', MultinomialNB(alpha=0.001))]) # 0.5459
        
clf = clf_nb
clf.fit(cleaned_train_data_txts, y_train_txt)
y_pred = clf.predict_proba(cleaned_test_data)

indexes =[]
for k,v in enumerate(test_texts):
    indexes.append(len(test_texts[v]))
    

y_pred_aggr = np.zeros((len(test_hosts), 8))

n, _ = np.shape(y_pred_aggr)
p = 0
q = 0
for i in range(n):
    q = p + indexes[i]
    y_pred_aggr[i,:] = np.mean(y_pred[p : q, :], axis= 0)
    p = q
    
    


# Write predictions to a file
with open('baseline_aggr_texts.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i,test_host in enumerate(test_hosts):
        lst = y_pred_aggr[i,:].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)