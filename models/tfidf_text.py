import string
import csv
import sys
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.append('../')
from preprocess import remove_duplicates, import_texts, generate_data, clean_host_texts

data = '../data/'
train_file = data + 'train.csv'
train_hosts, y_train = remove_duplicates(train_file)
texts_path = '../text/text'
texts = import_texts(texts_path)

with open(data + 'test.csv', 'r') as f:
    test_hosts = f.read().splitlines()
    
train_data = generate_data(train_hosts, texts)
test_data = generate_data(test_hosts, texts)

# Preprocessing texts
tokenizer = TweetTokenizer()
punctuation = string.punctuation + '’“”.»«…'
stpwords_fr = stopwords.words('french')
stpwords_en = stopwords.words('english')
cleaned_train_data = clean_host_texts(data=train_data, tok=tokenizer, 
                                      stpwds=stpwords_fr + stpwords_en, punct=punctuation)
cleaned_test_data = clean_host_texts(data=test_data, tok=tokenizer, 
                                     stpwds=stpwords_fr + stpwords_en, punct=punctuation)

# Pipeline
clf_lgr = Pipeline([
    ('vect', TfidfVectorizer(decode_error='ignore', sublinear_tf=True,
                             min_df=0.06, max_df=0.9)),
    ('clf', LogisticRegression(solver='lbfgs', multi_class='auto', 
                               max_iter=100, C=4))])
        
# Evaluate the model
X_train, X_eval, Y_train, Y_eval = train_test_split(
    cleaned_train_data, y_train, test_size=0.2, random_state=42
)
clf_lgr.fit(X_train, Y_train)
print("Classifier score: ", clf_lgr.score(X_eval, Y_eval))
print("Classifier multiclass loss: ", log_loss(Y_eval, clf_lgr.predict_proba(X_eval)))

# Choosing the model to make the predictions
clf = Pipeline([
    ('vect', TfidfVectorizer(decode_error='ignore', sublinear_tf=True,
                             min_df=0.06, max_df=0.9)),
    ('clf', LogisticRegression(solver='lbfgs', multi_class='auto', 
                               max_iter=100, C=4))])

# Choosing classifier
clf.fit(cleaned_train_data, y_train)
y_pred = clf.predict_proba(cleaned_test_data)

# Write predictions to a file
with open('../tfidf_text.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
