from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.model_selection import cross_val_score
import string
import numpy as np
import sys
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.append('../')
from preprocess import get_train_data, import_texts, generate_data, clean_host_texts


data = '../data/'
train_file = data + 'train.csv'
train_hosts, y_train = get_train_data(train_file)
texts_path = '../text/text'
texts = import_texts(texts_path)

with open(data + 'test.csv', 'r') as f:
    test_hosts = f.read().splitlines()

train_data = generate_data(train_hosts, texts)

# Preprocessing texts
tokenizer = TweetTokenizer()
punctuation = string.punctuation + '’“”.»«…°'
stpwords_fr = stopwords.words('french')
stpwords_en = stopwords.words('english')
cleaned_train_data = clean_host_texts(data=train_data, tok=tokenizer,
                                      stpwds=stpwords_fr + stpwords_en, punct=punctuation)

dict_y = dict([(j, i+2) for (i, j) in enumerate(set(y_train))])
y = [dict_y[x] for x in y_train]

# Logistic Regression Model
clas = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', sublinear_tf=True)),
                 ('clf', LogisticRegression(solver='lbfgs', multi_class='auto'))])

# The list of hyper-parameters we want to optimize. For each one we define the bounds,
# the corresponding scikit-learn parameter name, as well as how to sample values
# from that dimension ('log-uniform' for the learning rate)
space = [
    Categorical(categories=[(1, 1), (1, 2), (2, 2)], name='vect__ngram_range'),
    Real(0.3, 1.0, name='vect__max_df'),
    Real(0.0, 0.2, name='vect__min_df'),
    Categorical(categories=[True, False], name='vect__binary'),
    Categorical(categories=['l1', 'l2'], name='vect__norm'),
    Categorical(categories=[True, False], name='vect__smooth_idf'),
    Real(1e-4, 1e-2, name='clf__tol'),
    Real(0.3, 1, name='clf__C'),
    Categorical([None, 'balanced'], name='clf__class_weight'),
    Integer(100, 1000, name='clf__max_iter')
]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    clas.set_params(**params)
    return -np.mean(cross_val_score(clas, cleaned_train_data, y, cv=5, n_jobs=-1,
                                    scoring="neg_log_loss"))


res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)

print("Best score: ", res_gp.fun)
print("Best parameters: ", res_gp.x)
