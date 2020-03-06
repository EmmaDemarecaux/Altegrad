import csv
import string
import networkx as nx
import numpy as np
import sys
sys.path.append('../')
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from utils_deepwalk import deepwalk
from preprocess import remove_duplicates, import_texts, generate_data, clean_host_texts


data = '../data/'
train_file = data + 'train.csv'
train_hosts, y_train = remove_duplicates(train_file)
texts_path = '../text/text'
texts = import_texts(texts_path)

# Train data
with open(data + 'test.csv', 'r') as f:
    test_hosts = f.read().splitlines()
    
train_data = generate_data(train_hosts, texts)
test_data = generate_data(test_hosts, texts)

# Preprocessing texts
tokenizer = TweetTokenizer()
punctuation = string.punctuation + '’“”.»«…'
stpwords = stopwords.words('french')
cleaned_train_data = clean_host_texts(data=train_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)
cleaned_test_data = clean_host_texts(data=test_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)


# Model Logistic regression
clf_lgr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)

# TF-IDF
tf = TfidfVectorizer(decode_error='ignore', min_df=0.03, max_df=0.5, 
                     sublinear_tf=True, smooth_idf=True)
x_train_or = tf.fit_transform(cleaned_train_data)
x_train_or = x_train_or.toarray()

# Graph
G = nx.read_weighted_edgelist('../edgelist.txt', create_using=nx.DiGraph())
n = len(train_hosts)
n_dim = 128
n_walks = 500
walk_length = 100
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((G.number_of_nodes(), n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i, :] = model.wv[str(node)]

x_train = np.zeros((x_train_or.shape[0], x_train_or.shape[1]+n_dim))
for i in range(x_train_or.shape[0]):
    x_train[i] = np.concatenate((x_train_or[i], embeddings[int(train_hosts[i])]))

# Evaluate the model
X_train, X_eval, Y_train, Y_eval = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

clf_lgr.fit(X_train, Y_train)
print("Classifier score: ", clf_lgr.score(X_eval, Y_eval))
print("Classifier multiclass loss: ", log_loss(Y_eval, clf_lgr.predict_proba(X_eval)))


# Test data
x_test_or = tf.transform(cleaned_test_data)   
x_test_or = x_test_or .toarray()
x_test = np.zeros((x_test_or.shape[0], x_test_or.shape[1]+n_dim))
for i in range(x_test_or.shape[0]):
    x_test[i] = np.concatenate((x_test_or[i], embeddings[int(test_hosts[i])]))
    
# Choosing classifier
clf = clf_lgr
clf.fit(x_train, y_train)
y_pred = clf.predict_proba(x_test)

# Write predictions to a file
with open('../tfidf_deepwalk.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
