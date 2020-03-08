### Adavance Learning for Text and Graph Data (Altegrad 2020)
>**Khaoula BELAHSEN - Aicha BOUJANDAR - Emma DEMARECAUX**
************************
>*Kaggle Challenge: French Web Domain Classification*

## Folder 1: data 
It contains the challenge data that was provided: `edgelist.txt`, `train.csv` and `test.csv`.

## Folder 2: texts 
It contains the folder `text` which contains all the web domains texts.

## Folder 3: models 
Each `.py` file contained in this folder represents a model. Here is a brief description of the files models (the models are detailed in the report):
* `graph baseline.py`: the graph model that was initially provided.
* `text_baseline.py`: the text model model that was initially provided.
* `tfidf_deepwalk.py`: apply a _TF-IDF_ transformation on the text data and the _deepwalk_ method (contained in `utils_deepwalk.py`) to the graph data.
* `tfidf_gnn.py`: apply a _TF-IDF_ transformation on the text data and the _GNN_ method (contained in `utils_gnn.py`) on the graph data.
* `tfidf_text.py`: apply a _TF-IDF_ transformation on the text data followed by a _Logistic Regression Model_.

To generate the prediction file corresonding to a model, one needs to run the corresponding `.py` file as follows:

1. Change the current working directory to models:
```
cd models
```
2. Run the model.py file using the command: 
```
python3 model.py
```
3. The `.csv` prediction file will be saved in the root directory.

## Folder 4 : utils 
It contains the following util files:

* `utils_deepwalk.py`: contains the util functions to perform the _deepwalk_ algortihm.
* `utils_gnn.py`: contains the util functions to perform a _GNN_.
* `tfidf_text_param_opt.py`: contains the code to optimize the parameters of the _TF-IDF_ - _Logistic Regression_ model using skopt bayesian optimisation.

Finally, the file `preprocess.py` in the root directory contains the preprocessing steps appied on the text data and used by all the nodels.
