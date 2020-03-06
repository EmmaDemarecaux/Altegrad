import os
import csv
import codecs
import string
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TweetTokenizer
sys.path.append('../')
from preprocess import remove_duplicates, clean_host_texts


data = '../data/'
train_file = data + 'train.csv'
train_hosts, y_train = remove_duplicates(train_file)

# Load the textual content of a set of webpages for each host into the dictionary "text".
# The encoding parameter is required since the majority of our text is french.
filenames = os.listdir('../text/text')
splittingtxt = '__________________________________________________________________'
filenameformat = '#.txt'


def newfout(inputfile, filenum):
    filename = '../text_data/' + inputfile + filenameformat.replace('#', '_' + str(filenum))
    fout = open(filename, 'w')
    return fout


def split_train(filenames, splittingtxt, train_hosts):
    texts = dict()
    y_train_txt = list()  
    for filename in filenames:
        if filename in train_hosts:
            texts[filename] = []
            file = open(os.path.join('../text/text/', filename), errors='ignore')
            lines = file.readlines()
            filenum = 1
            fout = newfout(filename, filenum)

            for line in lines:
                if splittingtxt in line:
                    fout.close()
                    with codecs.open(os.path.join('../text_data/', filename + filenameformat.replace('#', '_' + str(filenum))),
                                     encoding="utf8", errors='ignore') as ff:
                        texts[filename].append(ff.read().replace("\n", "").lower())
                    filenum += 1
                    fout = newfout(filename, filenum)
                else:
                    fout.write(line)
            fout.close()
            with codecs.open(os.path.join('../text_data/', filename + filenameformat.replace('#', '_' + str(filenum))),
                             encoding="utf8", errors='ignore') as ff:
                texts[filename].append(ff.read().replace("\n", "").lower())
            for _ in range(filenum):
                y_train_txt.append(y_train[filename])
    return texts, y_train_txt


def split_test(filenames, splittingtxt, test_hosts):
    texts = dict()
    for filename in filenames : 
        if filename in test_hosts : 
            texts[filename] = []
            file = open(os.path.join('../text/text/', filename), errors='ignore')
            lines = file.readlines()
            filenum = 1
            fout = newfout(filename, filenum)

            for line in lines:
                if splittingtxt in line:
                    fout.close()
                    with codecs.open(os.path.join('../text_data/', filename + filenameformat.replace('#', '_' + str(filenum))),
                                     encoding="utf8", errors='ignore') as ff:
                        texts[filename].append(ff.read().replace("\n", "").lower())
                    filenum += 1
                    fout = newfout(filename, filenum)
                else:
                    fout.write(line)
            fout.close()
            with codecs.open(os.path.join('../text_data/', filename + filenameformat.replace('#', '_' + str(filenum))),
                             encoding="utf8", errors='ignore') as ff:
                texts[filename].append(ff.read().replace("\n", "").lower())
    return texts


texts, y_train_txt = split_train(filenames, splittingtxt, train_hosts)

train_data_list = list()
for k, v in enumerate(texts):
    train_data_list.append(texts[v])

train_data_flat = []
for sublist in train_data_list:
    for item in sublist:
        train_data_flat.append(item)

# Preprocessing texts
tokenizer = TweetTokenizer()
punctuation = string.punctuation + '’“”.»«…'
stpwords_fr = stopwords.words('french')
stpwords_en = stopwords.words('english')
cleaned_train_data_txts = clean_host_texts(data=train_data_flat, tok=tokenizer,
                                           stpwds=stpwords_fr + stpwords_en, punct=punctuation)

# Tests sets
with open(data + "test.csv", 'r') as f:
    test_hosts = f.read().splitlines()
    
test_texts = split_test(filenames, splittingtxt, test_hosts)

test_data_list = list()
for k, v in enumerate(test_texts):
    test_data_list.append(test_texts[v])


test_data_flat = []
for sublist in test_data_list:
    for item in sublist:
        test_data_flat.append(item)
        
cleaned_test_data = clean_host_texts(data=test_data_flat, tok=tokenizer,
                                     stpwds=stpwords_fr + stpwords_en, punct=punctuation)

# Logistic Regression Model
clf_lgr = Pipeline([
    ('vect', TfidfVectorizer(decode_error='ignore', sublinear_tf=True,
                             min_df=0.06, max_df=0.9)),
    ('clf', LogisticRegression(solver='lbfgs', multi_class='auto',
                               max_iter=100, C=4))])
clf = clf_lgr
clf.fit(cleaned_train_data_txts, y_train_txt)
y_pred = clf.predict_proba(cleaned_test_data)

indexes = []
for k, v in enumerate(test_texts):
    indexes.append(len(test_texts[v]))

y_pred_aggr = np.zeros((len(test_hosts), 8))

n, _ = np.shape(y_pred_aggr)
p = 0
q = 0
for i in range(n):
    q = p + indexes[i]
    y_pred_aggr[i, :] = np.mean(y_pred[p:q, :], axis=0)
    p = q

# Write predictions to a file
with open('../separate_texts.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i,test_host in enumerate(test_hosts):
        lst = y_pred_aggr[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
