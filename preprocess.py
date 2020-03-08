"""
# Preprocessing functions to be used for the different methods afterwards 
"""
import os
import codecs
from bs4 import BeautifulSoup
import re
from nltk.stem.snowball import FrenchStemmer

# Removing the text of the nodes which present some text errors such as:
page_errors = ['the document has moved here',
               'the requested url was rejected',
               'you need to enable javascript to run this app',
               '/tools/ disallow',
               '(button)    (button)    (button)    (button)    (button)    (button)    (button)    (button)',
               '301 moved permanently',
               '403 forbidden',
               'the requested url was not found on this server',
               'not found   the requested url',
               'was not found on this server.',
               'des corrections sont nécessaires pour sa réouverture',
               'plus disponible suite à un incident',
               'you are being redirected',
               'object moved permanently',
               "forbidden   you don't have permission",
               'refresh(0 sec)',
               'error 403 go away',
               'varnish cache server',
               'go away  guru meditation',
               'unable to complete your request',
               'internal server error',
               'the server encountered an internal error',
               'frame: ort   click here',
               'object moved   this document may be found here',
               'moved permanently.   moved permanently.',
               'ce document peut être consulté ici',
               '\ufeff   [page%20parking.jpg]',
               'ce site requiert un navigateur de version récente',
               "page d'erreur",
               '302 found',
               'click here if you are not automatically redirected'
               'redirect (policy_redirect)',
               'contact your network support team',
               'please enable js and disable any ad blocker',
               'object moved to here',
               'object moved   here',
               'the document has been permanently moved',
               '301moved permanently',
               'nescafé dolce gusto',
               'sign in   loading…',
               'tapez ici vos mots c submit',
               'continue on this website you will be providing your consent to our use'
               ]


# Function to detect text errors
def is_page_error(text, errors):
    for error in errors:
        if error in text:
            return True
    return False
    

# main preprocessing function
def clean_host_texts(data, tok, stpwds, punct, verbosity=5):
    cleaned_data = []
    for counter, host_text in enumerate(data):
        # converting article to lowercase already done
        temp = BeautifulSoup(host_text, 'lxml')
        text = temp.get_text()  # removing HTML formatting
        text = text.replace('{html}', "")  # removing "html"
        text = re.sub(r'http\S+', '', text)  # removing any url
        text = re.sub(r'\S*@\S*\s?', '', text)  # removing any e-mail address
        text = re.sub(r'\[\S*.\S*\]', '', text)  # removing any comment or image between square brackets
        text = re.sub(r'[^\s\.]+\>', '', text)  # removing any string of the form string>
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
    return [re.sub(r'\b(\w+)( \1\b)+', r'\1', ' '.join(l for l in sub_cleaned_data)) for 
            sub_cleaned_data in cleaned_data]


# Getting train hosts and labels
def get_train_data(train_file):
    with open(train_file, 'r') as f:
        train_data_ids = f.read().splitlines()
    train_hosts = list()
    y_train = list()
    for row in train_data_ids:
        host, label = row.split(",")
        train_hosts.append(host)
        y_train.append(label.lower())
    return train_hosts, y_train 


# Getting text data
def import_texts(texts_path):
    texts = dict()
    file_names = os.listdir(texts_path)
    for filename in file_names:
        with codecs.open(os.path.join(texts_path, filename), encoding='utf8', errors='ignore') as f:
            texts[filename] = f.read().replace('\n', '').lower()
    return texts
            

def generate_data(data_hosts, texts, remove_page_errors=True):
    """Get textual content of web hosts of the training set"""
    data = list()
    for host in data_hosts:
        if host in texts:
            to_write = texts[host]
            # removing text with errors
            if remove_page_errors and is_page_error(to_write, page_errors) and len(to_write) < 2000:
                data.append('')
            else:
                data.append(to_write)
        else:
            data.append('')  
    return data
