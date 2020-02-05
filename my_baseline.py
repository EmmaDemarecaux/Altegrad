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


def clean_host_texts(data, tokenizer, stpwds, punct, verbosity=5):
    cleaned_data = []
    for counter, host_text in enumerate(data):
        host_text = host_text.lower() # converting article to lowercase
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

# Read test data
with open("test.csv", 'r') as f:
    test_hosts = f.read().splitlines()

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

# Get textual content of web hosts of the test set
test_data = list()
for host in test_hosts:
    if host in texts:
        test_data.append(texts[host])
    else:
        test_data.append('')
        
cleaned_test_data = clean_host_texts(data=test_data, tokenizer=tokenizer, 
                                     stpwds=stpwds, punct=punct)

# Create the training matrix. Each row corresponds to a web host and each column to a word present in at least 10 web
# hosts and at most 1000 web hosts. The value of each entry in a row is equal to the tf-idf weight of that word in the 
# corresponding web host       

#vec = TfidfVectorizer(decode_error='ignore', strip_accents='unicode', encoding='latin-1', min_df=10, max_df=1000)
#X_train = vec.fit_transform(train_data)
vec = TfidfVectorizer(decode_error='ignore', min_df=10, max_df=100)
X_train = vec.fit_transform(cleaned_train_data)
        
# Create the test matrix following the same approach as in the case of the training matrix
X_test = vec.transform(cleaned_test_data)

print("Train matrix dimensionality: ", X_train.shape)
print("Test matrix dimensionality: ", X_test.shape)

# Use logistic regression to classify the webpages of the test set
clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))

y_pred = clf.predict_proba(X_test)

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
        