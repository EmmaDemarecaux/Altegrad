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
import pickle
import collections
from sklearn.metrics import log_loss
import numpy as np
np.random.seed(42)


def clean_host_texts(data, tok, stpwds, punct, verbosity=5):
    cleaned_data = []
    for counter, host_text in enumerate(data):
        # converting article to lowercase already done
        temp = BeautifulSoup(host_text, 'lxml')
        text = temp.get_text()  # removing HTML formatting
        text = text.replace('{html}',"") 
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
    

# Read training data: hosts + labels
with open("train.csv", 'r') as f:
    train_data = f.read().splitlines()

train_data = list(set(train_data))
duplicates_to_drop = [item for item, count in 
                      collections.Counter([item.split(",")[0] for item 
                                           in train_data]).items() if count > 1]

# Remove duplicates in training data: hosts + labels
train_hosts = list()
y_train = list()
for row in train_data:
    host, label = row.split(",")
    if host not in duplicates_to_drop:
        train_hosts.append(host)
        y_train.append(label.lower())


# Load the textual content of a set of web-pages for each host into the dictionary "text".
# The encoding parameter is required since the majority of our text is french.
texts = dict()
file_names = os.listdir('text/text')
for filename in file_names:
    with codecs.open(os.path.join('text/text/', filename), encoding="utf8", errors='ignore') as f: 
        texts[filename] = f.read().replace("\n", "").lower()


# Get textual content of web hosts of the training set
train_data = list()
for host in train_hosts:
    if host in texts:
        train_data.append(texts[host])
    else:
        train_data.append('')


# Preprocessing texts
tokenizer = TweetTokenizer()
punctuation = string.punctuation + '’“”.»«…'
stpwords = stopwords.words('french')
cleaned_train_data = clean_host_texts(data=train_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)

## Saving cleaned_train_data
#with open('cleaned_train_data.pkl', 'wb') as f:
#    pickle.dump(cleaned_train_data, f)
#    
## Loading cleaned_train_data
#with open('cleaned_train_data.pkl', 'rb') as f:
#    cleaned_train_data = pickle.load(f)


# LGR
#clf_lgr = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', min_df=0.03, max_df=0.8,
#                                             sublinear_tf=True)),
#                    ('clf', LogisticRegression(solver='lbfgs',
#                                               multi_class='auto', max_iter=1000))])
        
#clf_lgr = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', 
#                                             ngram_range=(1, 2),
#                                             max_df=0.8,
#                                             min_df=0.085,
#                                             smooth_idf=False,
#                                             sublinear_tf=True)),
#                    ('clf', LogisticRegression(solver='lbfgs',
#                                               multi_class='auto', 
#                                               tol=0.006,
#                                               max_iter=500))])
# best baseline for now     
#0.58656330749354
#1.2040824426316163

import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from preprocess import * 
import numpy as np


train_file = 'train.csv'
train_hosts, y_train = remove_duplicates(train_file)
texts_path = 'text/text'
texts = import_texts(texts_path)

with open("test.csv", 'r') as f:
    test_hosts = f.read().splitlines()
    
train_data = generate_data(train_hosts, texts)
test_data = generate_data(test_hosts, texts)

# Preprocessing texts
tokenizer = TweetTokenizer()
punctuation = string.punctuation + '’“”.»«…'
stpwords = stopwords.words('french')

cleaned_train_data = clean_host_texts(data=train_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)
cleaned_test_data = clean_host_texts(data=test_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)



clf_lgr = Pipeline([('vect', TfidfVectorizer(decode_error='ignore',
                                             max_df=0.3,
                                             min_df=0.078,
                                             smooth_idf=False,
                                             sublinear_tf=True)),
                    ('clf', LogisticRegression(solver='lbfgs',
                                               multi_class='auto', 
                                               tol=0.01,
                                               max_iter=500))])
        
#[(1, 1), 0.3, 0.07844280113310775, False, 'l2', False, 0.01, 1.0, None, 100]


# Evaluate the model
X_train, X_eval, Y_train, Y_eval = train_test_split(
        cleaned_train_data, y_train, test_size=0.2, random_state=42
        )
clf_lgr.fit(X_train, Y_train)
print(clf_lgr.score(X_eval, Y_eval))
print(log_loss(Y_eval, clf_lgr.predict_proba(X_eval)))



import skopt
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score, KFold

dic = dict([(j,i+2) for (i,j) in enumerate(set(y_train))])
y = [dic[x] for x in y_train]

clas = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', sublinear_tf=True)),
                 ('clf', LogisticRegression(solver='lbfgs', multi_class='auto'))])

# The list of hyper-parameters we want to optimize. For each one we define the bounds,
# the corresponding scikit-learn parameter name, as well as how to sample values
# from that dimension ('log-uniform' for the learning rate)
space  = [Categorical(categories = [(1,1), (1,2), (2,2)], name='vect__ngram_range'),
          Real(0.3, 1.0, name='vect__max_df'),
          Real(0.0, 0.2, name='vect__min_df'),
          Categorical(categories = [True, False], name='vect__binary'),
          Categorical(categories = ['l1', 'l2'], name='vect__norm'),
          Categorical(categories = [True, False], name='vect__smooth_idf'),
          Real(1e-4, 1e-2, name='clf__tol'),
          Real(0.3, 1, name='clf__C'),
          Categorical([None, 'balanced'], name ='clf__class_weight'),
          Integer(100, 1000, name = 'clf__max_iter')
          ]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    clas.set_params(**params)

    return -np.mean(cross_val_score(clas, cleaned_train_data, y, cv=5, n_jobs=-1,
                                    scoring="neg_log_loss"))

from skopt import gp_minimize
res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)


print(res_gp.fun)
print(res_gp.x)



# Test data
# Read test data
with open("test.csv", 'r') as f:
    test_hosts = f.read().splitlines()
    
# Get textual content of web hosts of the test set
test_data = list()
for host in test_hosts:
    if host in texts:
        test_data.append(texts[host])
    else:
        test_data.append('')

# Preprocessing texts
cleaned_test_data = clean_host_texts(data=test_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)

## Saving cleaned_test_data
#with open('cleaned_test_data.pkl', 'wb') as f:
#    pickle.dump(cleaned_test_data, f)
#    
## Loading cleaned_test_data
#with open('cleaned_test_data.pkl', 'rb') as f:
#    cleaned_test_data = pickle.load(f)
    
# Choosing classifier
clf = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', 
                                             ngram_range=(1, 2),
                                             max_df=0.8,
                                             min_df=0.085,
                                             smooth_idf=False,
                                             sublinear_tf=True)),
                    ('clf', LogisticRegression(solver='lbfgs',
                                               multi_class='auto', 
                                               tol=0.006,
                                               max_iter=500))])
clf.fit(cleaned_train_data, y_train)
y_pred = clf.predict_proba(cleaned_test_data)

# Write predictions to a file
with open('best_baseline.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
