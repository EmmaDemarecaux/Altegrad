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


# Read training data
with open("train.csv", 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
y_train = list()
for row in train_data:
    host, label = row.split(",")
    train_hosts.append(host)
    y_train.append(label.lower())

# Load the textual content of a set of webpages for each host into the dictionary "text". 
# The encoding parameter is required since the majority of our text is french.
texts = dict()
filenames = os.listdir('text/text')
for filename in filenames:
    with codecs.open(os.path.join('text/text/', filename), encoding="utf8", errors='ignore') as f: 
        texts[filename] = f.read().replace("\n", "").lower()

train_data = list()
for host in train_hosts:
    if host in texts:
        train_data.append(texts[host])
    else:
        train_data.append('')


tokenizer = TweetTokenizer()
punct = string.punctuation.replace('-', '’“”.»«')
stpwds = stopwords.words('french')
cleaned_train_data = clean_host_texts(data=train_data, tokenizer=tokenizer, 
                                     stpwds=stpwds, punct=punct)

X_train, X_eval, Y_train, Y_eval = train_test_split(
        cleaned_train_data, y_train, test_size=0.2, random_state=42
        )    

# ========================= SVM don't have predict_proba
clf_svm = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', ngram_range=(1, 2))),
                    ('clf', SGDClassifier(alpha=0.0001, max_iter=1000, 
                                          penalty='l2', random_state=42))])
     
clf_svm.fit(X_train, Y_train)
print(clf_svm.score(X_eval, Y_eval))  
        
# ========================= NB

clf_nb = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', ngram_range=(1, 2))),
                    ('clf', MultinomialNB(alpha=0.001))]) # 0.5459

# ----------------- Grid Search
#parameters = {
#        'clf__alpha': (1.0, 1e-1, 1e-2, 1e-3, 1e-4),
#        }
#gs_clf = GridSearchCV(clf_nb, parameters, n_jobs=-1)
#gs_clf = gs_clf.fit(cleaned_train_data, y_train)
#print(gs_clf.best_score_)
#print(gs_clf.best_params_)

clf_nb.fit(X_train, Y_train)
print(clf_nb.score(X_eval, Y_eval))

# ========================= LGR

clf_lgr = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', min_df=10, max_df=1000)),
                    ('clf', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000))])

clf_lgr.fit(X_train, Y_train)
print(clf_lgr.score(X_eval, Y_eval)) 

# ========================= GB

#clf_gb = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', ngram_range=(1, 2))),
#                    ('clf', GradientBoostingClassifier())])

# ------------------------- Grid Search
#parameters = {
#        'clf__alpha': (1.0, 1e-1, 1e-2, 1e-3, 1e-4),
#        }
#gs_clf = GridSearchCV(clf_nb, parameters, n_jobs=-1)
#gs_clf = gs_clf.fit(cleaned_train_data, y_train)
#print(gs_clf.best_score_)
#print(gs_clf.best_params_)

#clf_gb.fit(X_train, Y_train)
#print(clf_gb.score(X_eval, Y_eval))  

# =========================----------- Test data

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
        
cleaned_test_data = clean_host_texts(data=test_data, tokenizer=tokenizer, 
                                     stpwds=stpwds, punct=punct)

# ========================= Choosing classifier

clf = clf_nb
clf.fit(cleaned_train_data, y_train)
y_pred = clf.predict_proba(cleaned_test_data)

# Write predictions to a file
with open('my_baseline.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i,test_host in enumerate(test_hosts):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
        