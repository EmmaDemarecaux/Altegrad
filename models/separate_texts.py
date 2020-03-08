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
from preprocess import get_train_data, clean_host_texts

data = '../data/'
train_file = data + 'train.csv'
train_hosts, y_train = get_train_data(train_file)

# Loading the textual content of a set of web pages for each host into the dictionary "text".
# The encoding parameter is required since the majority of our text is french.
file_names = os.listdir('../text/text')
splitting_text = '__________________________________________________________________'
file_name_format = '#.txt'


def new_f_out(input_file, file_num):
    """
    Function to create the file for each subtext of the original bigger text 

    Input : 
        - input_file : the file name str for the text which we seperate 
        - file_num : index for the subtext 
    Output :
        - f_out : file where to write the subtext  
    """
    filename = '../text_data/' + input_file + file_name_format.replace('#', '_' + str(file_num))
    f_out = open(filename, 'w')
    return f_out


def split_train(file_names, splitting_text, train_hosts):
    """
    Function to create the enlarged training sets for subtexts and the labels 

    Input : 
        - file_names : path of the texts  
        - splitting : splitting criterion
        - train_hosts : the original texts 
    Output :
        - texts : the subtexts for training 
        - y_train_text : their labels  
    """
    texts = dict()
    y_train_text = list()  
    for filename in file_names:
        if filename in train_hosts:
            texts[filename] = []
            file = open(os.path.join('../text/text/', filename), errors='ignore')
            lines = file.readlines()
            file_num = 1
            f_out = new_f_out(filename, file_num)

            for line in lines:
                if splitting_text in line:
                    f_out.close()
                    with codecs.open(os.path.join('../text_data/', filename + file_name_format.replace('#', '_' + str(file_num))),
                                     encoding="utf8", errors='ignore') as ff:
                        texts[filename].append(ff.read().replace("\n", "").lower())
                    file_num += 1
                    f_out = new_f_out(filename, file_num)
                else:
                    f_out.write(line)
            f_out.close()
            with codecs.open(os.path.join('../text_data/', filename + file_name_format.replace('#', '_' + str(file_num))),
                             encoding="utf8", errors='ignore') as ff:
                texts[filename].append(ff.read().replace("\n", "").lower())
            for _ in range(file_num):
                y_train_text.append(y_train[filename])
    return texts, y_train_text


def split_test(file_names, splitting_text, test_hosts):
    """
    Does the same as split_train but on the test texts  
    
    """
    texts = dict()
    for filename in file_names:
        if filename in test_hosts:
            texts[filename] = []
            file = open(os.path.join('../text/text/', filename), errors='ignore')
            lines = file.readlines()
            file_num = 1
            f_out = new_f_out(filename, file_num)

            for line in lines:
                if splitting_text in line:
                    f_out.close()
                    with codecs.open(os.path.join('../text_data/', filename + file_name_format.replace('#', '_' + str(file_num))),
                                     encoding="utf8", errors='ignore') as ff:
                        texts[filename].append(ff.read().replace("\n", "").lower())
                    file_num += 1
                    f_out = new_f_out(filename, file_num)
                else:
                    f_out.write(line)
            f_out.close()
            with codecs.open(os.path.join('../text_data/', filename + file_name_format.replace('#', '_' + str(file_num))),
                             encoding="utf8", errors='ignore') as ff:
                texts[filename].append(ff.read().replace("\n", "").lower())
    return texts


# Train sets
texts, y_train_txt = split_train(file_names, splitting_text, train_hosts)

train_data_list = list()
for k, v in enumerate(texts):
    train_data_list.append(texts[v])

train_data_flat = []
for sublist in train_data_list:
    for item in sublist:
        train_data_flat.append(item)

# Tests sets
with open(data + "test.csv", 'r') as f:
    test_hosts = f.read().splitlines()
    
test_texts = split_test(file_names, splitting_text, test_hosts)

test_data_list = list()
for k, v in enumerate(test_texts):
    test_data_list.append(test_texts[v])

test_data_flat = []
for sublist in test_data_list:
    for item in sublist:
        test_data_flat.append(item)

# Preprocessing texts
tokenizer = TweetTokenizer()
punctuation = string.punctuation + '’“”.»«…°'
stpwords_fr = stopwords.words('french')
stpwords_en = stopwords.words('english')
cleaned_train_data_texts = clean_host_texts(data=train_data_flat, tok=tokenizer,
                                            stpwds=stpwords_fr + stpwords_en, punct=punctuation)
cleaned_test_data = clean_host_texts(data=test_data_flat, tok=tokenizer,
                                     stpwds=stpwords_fr + stpwords_en, punct=punctuation)

# Logistic Regression Model
clf_lgr = Pipeline([
    ('vect', TfidfVectorizer(decode_error='ignore', sublinear_tf=True, ngram_range=(1, 1),
                             min_df=0.0149, max_df=0.9, binary=False, smooth_idf=True)),
    ('clf', LogisticRegression(tol=1e-05, C=4.59))])
clf = clf_lgr
clf.fit(cleaned_train_data_texts, y_train_txt)
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
    for i, test_host in enumerate(test_hosts):
        lst = y_pred_aggr[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
