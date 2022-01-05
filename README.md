# AC-HEN
For KBS‘22-submission
> Heterogeneous Graph Neural Network for Attribute Completion

## Dependencies
Recent versions of the following packages for Python 3 are required:
* dgl==0.5.3
* networkx==2.4
* node2vec==0.4.3
* numpy==1.17.2
* pandas==1.1.5
* scipy==1.54
* sikit-learn==0.21.3
* torch==1.7.1
* tqdm==4.62.3


## Datasets
The raw datasets are available at:
* IMDB https://github.com/cynricfu/MAGNN/tree/master/data/raw/IMDB
* DBLP https://github.com/cynricfu/MAGNN/tree/master/data/raw/DBLP  

The precessed data are avaliable in this project in the file DBLPdata , ACMdata and data.

## Run Code
The default dataset used in this project is IMDB.  
`python main.py`

If you want to change the default configuration, you can edit `loaddata` in `utils.py`. 

