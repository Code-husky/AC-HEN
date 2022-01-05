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
* IMDB https://www.dropbox.com/s/g0btk9ctr1es39x/IMDB_processed.zip?dl=0
* DBLP https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=0 
* ACM  https://pan.baidu.com/s/1w5rYW89OJl-rwuNgVftWMg password:pqw6
The precessed data are avaliable in this project in the file DBLPdata and ACMdata. The `data` file contains the process data of IMDB.

## Run Code
The default dataset used in this project is IMDB.  

Input this command in command line and run the code.  

`python main.py`

If you want to change the default configuration, you can edit `loaddata` in `utils.py` . For example, if you want to change dataset, you can use the data of 'IMDBdata' and 'ACMdata'.

