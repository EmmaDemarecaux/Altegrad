import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from preprocess import * 

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



cats, weights = np.unique(y_train, return_counts = True)
weights = weights / len(y_train)

dic_cat = dict(zip(cats, weights))
dic_cat

# LGR
clf_lgr = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', sublinear_tf=True,
                                             max_df = 0.7, min_df = 0.06, ngram_range = (1, 2),
                                             smooth_idf = True)),
                    ('clf', LogisticRegression(solver='lbfgs',
                                               multi_class='auto', class_weight= None,
                                               max_iter = 1000))])

# Evaluate the model
X_train, X_eval, Y_train, Y_eval = train_test_split(
        cleaned_train_data, y_train, test_size=0.2, random_state=42
        )

clf_lgr.fit(X_train, Y_train)
print(clf_lgr.score(X_eval, Y_eval))
print(log_loss(Y_eval, clf_lgr.predict_proba(X_eval)))


 
# Choosing classifier
clf = clf_lgr
clf.fit(cleaned_train_data, y_train)
y_pred = clf.predict_proba(cleaned_test_data)

# Write predictions to a file
with open('kb_baseline.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
