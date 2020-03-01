import csv
import json
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from keras.models import Model
from keras import backend as K
from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Concatenate, Dense

# = = = = = functions = = = = =

def visualize_doc_embeddings(my_doc_embs,my_colors,my_labels,my_name):
    my_pca = PCA(n_components=10)
    my_tsne = TSNE(n_components=2,perplexity=10) # https://lvdmaaten.github.io/tsne/
    doc_embs_pca = my_pca.fit_transform(my_doc_embs) 
    doc_embs_tsne = my_tsne.fit_transform(doc_embs_pca)
    
    fig, ax = plt.subplots()
    
    for label in list(set(my_labels)):
        idxs = [idx for idx,elt in enumerate(my_labels) if elt==label]
        ax.scatter(doc_embs_tsne[idxs,0], 
                   doc_embs_tsne[idxs,1], 
                   c = my_colors[label],
                   label=str(label),
                   alpha=0.7,
                   s=40)
    
    ax.legend(scatterpoints=1)
    fig.suptitle('t-SNE visualization doc embeddings',fontsize=15)
    fig.set_size_inches(11,7)
    fig.savefig(my_name + '.pdf')


# conv layers: https://keras.io/layers/convolutional/
# pooling layers: https://keras.io/layers/pooling/
def cnn_branch(n_filters,k_size,d_rate,my_input):
    return Dropout(d_rate)(GlobalMaxPooling1D()(Conv1D(filters=n_filters,
                                                       kernel_size=k_size,
                                                       activation='relu')(my_input)))

# = = = = = parameters = = = = =

mfw_idx = 2 # index of the most frequent words in the dictionary
padding_idx = 0
oov_idx = 1

d = 30 # dimensionality of word embeddings
max_size = 60 # max allowed size of a document
nb_branches = 2
nb_filters = 50
filter_sizes = [3,4]
drop_rate = 0.3 # amount of dropout regularization
batch_size = 64
nb_epochs = 1
my_optimizer = 'adam'

# = = = = = loading data = = = = =

# load dictionary of word indexes (sorted by decreasing frequency across the corpus)
with open('./data/word_to_index.json', 'r') as my_file:
    word_to_index = json.load(my_file)

# invert mapping (for sanity checking, later)
index_to_word = dict((v,k) for k,v in word_to_index.items())

with open('./data/training.csv', 'r') as my_file:
    reader = csv.reader(my_file, delimiter=',')
    x_train = list(reader)

with open('./data/test.csv', 'r') as my_file:
    reader = csv.reader(my_file, delimiter=',')
    x_test = list(reader)

with open('./data/training_labels.txt', 'r') as my_file:
    y_train = my_file.read().splitlines()

with open('./data/test_labels.txt', 'r') as my_file:
    y_test = my_file.read().splitlines()

# turn lists of strings into lists of integers
x_train = [[int(elt) for elt in sublist] for sublist in x_train]
x_test = [[int(elt) for elt in sublist] for sublist in x_test]  

y_train = [int(elt) for elt in y_train]
y_test = [int(elt) for elt in y_test]

print('data loaded')

# = = some sanity checking = =
    
print('index of "the":',word_to_index['the']) # most frequent word
print('index of "movie":',word_to_index['movie']) # very frequent word
print('index of "elephant":',word_to_index['elephant']) # less frequent word
    
# reconstruct first review
rev = x_train[0]
print (' '.join([index_to_word[elt] if elt in index_to_word else 'OOV' for elt in rev]))
# compare it with the original review: https://www.imdb.com/review/rw2219371/?ref_=tt_urv

# = = = = = truncation and padding = = = = =

# truncate reviews longer than 'max_size'
x_train = [rev[:max_size] for rev in x_train]
x_test = [rev[:max_size] for rev in x_test]

# pad reviews shorter than 'max_size' with the special padding token
x_train = [rev+[padding_idx]*(max_size-len(rev)) if len(rev)<max_size else rev for rev in x_train]
x_test = [rev+[padding_idx]*(max_size-len(rev)) if len(rev)<max_size else rev for rev in x_test]

# all reviews should now be of size 'max_size'
assert max_size == list(set([len(rev) for rev in x_train]))[0] and max_size == list(set([len(rev) for rev in x_test]))[0]

print('truncation and padding done')

# = = = = = defining architecture = = = = =

# see guide to Keras' functional API: https://keras.io/getting-started/functional-api-guide/
# core layers: https://keras.io/layers/core/

doc_ints = Input(shape=(None,))

doc_wv = Embedding(input_dim=len(word_to_index)+2, # vocab size + OOV token + padding token
                   output_dim=d, # dimensionality of embedding space
                   input_length=max_size, 
                   trainable=True
                   )(doc_ints)

doc_wv_dr = Dropout(drop_rate)(doc_wv)

branch_outputs = []
for idx in range(nb_branches):
    branch_outputs.append(cnn_branch(nb_filters,filter_sizes[idx],drop_rate,doc_wv_dr))

concat = Concatenate()(branch_outputs) # branch output combination

preds = Dense(units=1, 
              activation='sigmoid',
              )(concat)

model = Model(doc_ints,preds)

model.compile(loss='binary_crossentropy',
              optimizer = my_optimizer,
              metrics = ['accuracy'])

print('model compiled')

model.summary()

print('total number of model parameters:',model.count_params())

# = = = = = visualizing doc embeddings (before training) = = = = =

# you can access the layers of the model with model.layers
# then the input/output shape of each layer with, e.g., model.layers[0].input_shape or model.layers[0].output_shape

# extract output of the final embedding layer (before the softmax)
# in test mode, we should set the 'learning_phase' flag to 0 (e.g., we don't want to use dropout)
get_doc_embedding = K.function([model.layers[0].input,K.learning_phase()],
                               [model.layers[9].output])

n_plot = 1000
labels_plt = y_test[:n_plot]
doc_embs = get_doc_embedding([np.array(x_test[:n_plot]),0])[0]

print('plotting embeddings of first',n_plot,'documents')
visualize_doc_embeddings(doc_embs,['blue','red'],labels_plt,'before')

# = = = = = training = = = = =

model.fit(np.array(x_train[:2500]), 
          y_train[:2500],
          batch_size = batch_size,
          epochs = nb_epochs,
          validation_data = (np.array(x_test), y_test))

# = = = = = visualizing doc embeddings (after training) = = = = =
doc_embs = get_doc_embedding([np.array(x_test[:n_plot]),0])[0]
visualize_doc_embeddings(doc_embs,['blue','red'],labels_plt,'after')

# perform the same steps as before training and observe the changes

# = = = = = predictive text regions for the first branch = = = = =

get_region_embedding = K.function([model.layers[0].input,K.learning_phase()],
                                  [model.layers[3].output])

get_sigmoid = K.function([model.layers[0].input,K.learning_phase()],
                         [model.layers[10].output])

my_review_text = 'Oh , god , this was such a disappointment ! Worst movie ever . Not worth the 15 bucks .'
tokens = my_review_text.lower().split()
my_review = [word_to_index[elt] for elt in tokens]

# extract regions (sliding window over text)
regions = []
regions.append(' '.join(tokens[:filter_sizes[0]]))
for i in range(filter_sizes[0], len(tokens)):
    regions.append(' '.join(tokens[(i-filter_sizes[0]+1):(i+1)]))

my_review = np.array([my_review])

reg_emb = get_region_embedding([my_review,0])[0][0,:,:]

prediction = get_sigmoid([my_review,0])[0] # note: you can also use directly: predictions = model.predict(x_test[:100]).tolist()

norms = np.linalg.norm(reg_emb,axis=1) #[1:len(regions)]

print([list(zip(regions,norms))[idx] for idx in np.argsort(-norms).tolist()])

# = = = = = saliency map = = = = =

input_tensors = [model.input, K.learning_phase()]
saliency_input = model.layers[3].input # before convolution
saliency_output = model.layers[10].output # class score

gradients = model.optimizer.get_gradients(saliency_output,saliency_input)
compute_gradients = K.function(inputs=input_tensors,outputs=gradients)

matrix = compute_gradients([my_review,0])[0][0,:,:]

to_plot = np.absolute(matrix) #[:len(tokens),:]
#print(np.linalg.norm(matrix[:len(tokens),:],axis=1))

fig, ax = plt.subplots()
heatmap = ax.imshow(to_plot, cmap=plt.pyplot.cm.Blues, interpolation='nearest',aspect='auto')
ax.set_yticks(np.arange(len(tokens)))
ax.set_yticklabels(tokens)
ax.tick_params(axis='y', which='major', labelsize=32*10/len(tokens))
fig.colorbar(heatmap)
fig.set_size_inches(11,7)
fig.savefig('saliency_map.pdf',bbox_inches='tight')
fig.show()