### Adavance Learning for Text and Graph Data (Altegrad 2020)
>**Khaoula BELAHSEN - Aicha BOUJANDAR - Emma DEMARECAUX**
************************
>*Kaggle Challenge: French Web Domain Classification*

## Folder 1: data 
It contains the challenge data that was provided: `edgelist.txt`, `train.csv` and `test.csv`.

## Folder 2: texts 
It contains the folder text which contains all the web domains texts.

## Folder 3: models 
Each .py file contained in this file represents a model (except utils_deepwalk and utils_gnn). To generate the prediction file corresonding to a model, one needs to run the corresponding .py file. Here is a brief description of the files models (the models are detailed in the report):
* `graph baseline.py`: the graph model that was initially provided.
* `text_baseline.py`: the text model model that was initially provided.
* `tfidf_deepwalk.py`: apply a tfidf transformation on text data and the deepwalk method (contained in utils_deepwalk.py) to the graph data.
* `tfidf_gnn.py`: apply a tfidf transformation on text data and the gnn method (contained in utils_gnn.py) on the graph data.
* `tfidf_text.py`: apply a tfidf transformation on text data.

## Folder 4 : utils 
It contains the following util files:
* `utils_deepwalk.py`: contains the util functions to perform the deepwalk algortihm.
* `utils_gnn.py`: contains the util functions to perform a GNN.
* `tfidf_text_param_opt.py`: contains the code to optimize the parameters of the TF-IDF-Logistic-Regression model using skopt bayesian optimisation
