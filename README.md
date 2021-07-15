#Desc:
The goal of this project is to figure out if there is a correlation between the positive/negative sentiment of scientific papers (related to astronomy) and the number of citiations of a paper. In order to do this, we need to create a database of astronomy related papers in sqlite from the NASA ADS API. After creating this database, we need to preprocess the text of each paper, and extract sentiment analysis (unsupervised) of that paper. Afterwards, we need to compare the sentiment with the number of correlations based on the 
following measures: Pearson, Kendall, Spearman correlation.

#Database:

We collected papers from the NASA ADS from years 1980-2021. We are collecting papers from database:astronomy.

The following data fields were established:
- bibcode TEXT X
- alternate bibcode TEXT
- title TEXT X 
- date TEXT X
- year INTEGER X
- doctype TEXT X
- eid TEXT
- recid TEXT X
- esources TEXT X
- property TEXT X
- citation TEXT X
- read_count TEXT X
- author TEXT X 
- abstract TEXT X 
- citation_count INTEGER X
------
- file_path TEXT
- paper_cit_path TEXT 
- downloaded_pdf INTEGER
- ran_sentiment INTEGER
- sentiment REAL
- paper_text TEXT
- word_vector BLOB

#THINGS THAT NEED TO GET DONE: 

[] did not get abs value in stats for comp sentiment
[] figure out optimal cluster value for kmeans
[] figure out optimal number of components for lda
[] figure out how to get pdf data
    - only download pdf with eprint, pub_pdf not available w/0 link or access, not directly available
[] need to fix pdf obtaining
[] how to demonstrate code to professor
[] need to see if need to use dask or not for processing
    - if so, need to read dask tutorials and add dask into clustering
[] need to see if need to make pipeline for clustering and sentiment analysis   
    - if so, need to read tutorials
[] need to figure out how to update and insert at same time - cannot do            this, writing   will queue, locks for only a few milliseconds, can read mutliple times while writing
[] incrementally train data using partial fit from sklearn

------------------------
[] may need to make custom func to pass in to vectorizer
[] can also do sentiment analysis via neural network ????
[] using pytorch, has libary called skorch for sklearn ???
[] can use tensorflow, tensor flow is harder to learn but has tensorboard ???

#THINGS THAT HAVE BEEN DONE: 

[X] extracted 3 mil articles from NASA ADS - messed up
[X] performed K-means clustering 
[X] work on getting pdf database -- TAKING PAUSE ON THIS
[X] figure out how to replicate curl requests in 
[X] silouhette analysis shows that data is unstructured - do not draw any theoretical findings from it (SO)
[X] preprocessing of the data needs to be improved, not suitable for clustering - 
resolved, you have to use data2d to plot silouhette
[X] kmeans may be limiting, may not necessarily be preprocessing of data, try alternatives ? 
[X] try dbscan bc is good for big data - cancel, not good for text data
[X] get silouhette of both clusters - higher score is better cluster number
[X] get lda
[X] make sure manual and vect tokenization process is same
[X] can do sentiment analysis via clustering
[X] pos tagging
[X] topic modeling
[X] refactored text processing code
[X] refactored downloading code
[X] rewent through ADS to refigure out downloading code
[X] troubleshooting downloading code


------ text processing 

- once data has been downloaded:
go through text and process it placed processed text back into sqlite db