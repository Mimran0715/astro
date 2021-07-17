import warnings
warnings.filterwarnings("ignore")
import re 
import sqlite3 
import time
import os
import pandas as pd
import numpy as np
import sys
import pdfplumber as p

from nltk import FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from utils import get_data, db_command

def process_data(db_path, tbl_name, size, fields_long):
    #print("in function: get_data")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    #entry_count = db_command_2(db_path, conn, "SELECT COUNT(*) FROM " + str(tbl_name) + ";")

    #df = pd.read_sql_query("SELECT  * FROM " + str(tbl_name) + " LIMIT " + str(size) +  ";", conn)
    cur.execute("SELECT * FROM " + str(tbl_name) + " LIMIT " + str(size) + ";")
    rows = cur.fetchall()
    #print(rows[0])
    df = pd.DataFrame(rows, columns=fields_long)
    #print(df['abs_text'])
    df = process_text(df)
    #print()
    #print(df)
    conn.close()
    return df

def parse_text(text):
    return re.findall("[a-zA-Z0-9']{2,}", text)

def tokenize(text):
    return [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]

def tokenize_bigram(text):
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    return list(ngrams(tokens, 2))

def tokenize_and_stem(text):
    stemmer = SnowballStemmer('english')
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems  # taken from: http://brandonrose.org/clustering_mobile
    
def process_text(df):
    #stop_words = stopwords.words('english')
    abs_text = df['abstract']
    abs_text = abs_text.str.lower() 
    #print(abs_text.dtypes) 

    abs_text = abs_text.apply(str)
    abs_text = abs_text.apply(parse_text)
    abs_text = abs_text.apply(' '.join)
    #abs_text = abs_text.apply(stemmer.stem())

    abs_text_stem = abs_text.apply(tokenize_and_stem)
    abs_text_token = abs_text.apply(tokenize)
    abs_text_bigram = abs_text.apply(tokenize_bigram)
    #abs_text_trigram = abs_text.apply(tokenize_trigram)

    #print(abs_text_token)
    item_list = []
    for item in abs_text_token:
        item_list.append(" ".join(item))
    #print(item_list)
    df['abs_text'] = item_list
    #print(df['abs_text'])
    return df

def obtain_pdf_text(path:str):
    with p.open(path) as pdf:
        #print("hello")
        #print(pdf.pages)
        paper_text = ""
        t0 = time.time()
        for i in range(len(pdf.pages)):
            text = pdf.pages[i].extract_text(x_tolerance=1, y_tolerance=3)
            paper_text += text
    print(paper_text)
    print("Time taken to extract text", time.time() - t0)   
    return paper_text

# def db_command_2(path, conn, command, task):
#     #conn = sqlite3.connect(path)
#     c = conn.cursor()
#      #print(type(command), type(task))
#     if len(task) == 0:
#          c.execute(command)
#     else:
#         c.execute(command, task)
#     conn.commit()
#     conn.close()
#     return c.lastrowid

def main():
    #results_dir = "/Users/Mal/Documents/results/"
    #folder_dir = "/Users/Mal/Documents/results/kmeans/"
    db_path = '/Users/Mal/Documents/research.db'
    tbl_name = sys.argv[1]
    run_size = sys.argv[2] # size of entries to get from each year bracket
    #run_type = sys.argv[2] # kmeans == 0, lda == 1
    run_loc = sys.argv[3] # running on local == 0, running on clotho == 1

    if run_loc == 1:
        #results_dir = "/home/maleeha/research/results/"
        #folder_dir = "/Users/Mal/Documents/results/kmeans/"
        db_path = '/home/maleeha/research/research.db'

    # if(os.path.exists(results_dir)==0):
    #     os.mkdir(results_dir)

    # if(os.path.exists(folder_dir)==0):
    #     os.mkdir(folder_dir)
    # conn = sqlite3.connect(db_path)
    # entry_count = db_command_2(db_path, conn, "SELECT COUNT(*) FROM " + str(tbl_name) + ";")
    # conn.close()
    
    fields_long = ['bibcode', 'alternate_bibcode', 'title', 'date' , 'year', 'doctype', \
        'eid', 'recid', 'esources', 'property', 'citation', 'read_count', 'author', \
            'abstract', 'citation_count', 'identifier'] + ['file_path','downloaded_pdf', 'ran_sentiment', \
                'sentiment', 'paper_text', 'abs_text', 'paper_proc_text'] # + ['sql_id'] 

    df = process_data(db_path, tbl_name, int(run_size), fields_long)
    #df = process_text(df)
    
    conn = sqlite3.connect(db_path)
    #entry_count = db_command_2(db_path, conn, "SELECT COUNT(*) FROM " + str(tbl_name) + ";")
    #print(df.dtypes)
    df.to_sql(tbl_name, conn, if_exists='replace')  
    conn.close()
    
   
if __name__ == '__main__':
    main()

#from matplotlib.backends.backend_pdf import PdfPages
#import matplotlib.pyplot as plt
#import csv
#import pickle
#from wordcloud import WordCloud
#from sklearn.decomposition import TruncatedSVD
#from sklearn.cluster import MiniBatchKMeans
#plt.ion() # take away popups - putting plot images in pdf

# from sklearn.feature_extraction.text import TfidfVectorizer 
# from sklearn.feature_extraction.text import CountVectorizer 
# from sklearn.model_selection import train_test_split

# from collections import defaultdict
# from sklearn.decomposition import PCA
# from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

# from gensim.models import LsiModel
# from gensim.models import LdaModel
# from gensim.models.coherencemodel import CoherenceModel
# from gensim.matutils import Sparse2Corpus
# from gensim.corpora import Dictionary

# from sklearn.metrics import silhouette_samples, silhouette_score
# from yellowbrick.cluster import SilhouetteVisualizer

# import random
# import seaborn as sns 
# from pprint import pprint # lda link

#import pyLDAvis
#import pyLDAvis.gensim
#from scipy.cluster.hierarchy import dendrogram
#from scipy.cluster import hierarchy
#import pyLDAvis
#import pyLDAvis.sklearn
#pyLDAvis.enable_notebook()
#from sklearn.cluster import KMeans
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.cluster import AgglomerativeClustering
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
#from sklearn import feature_extraction # br
#from sklearn.decomposition import TruncatedSVD
#from sklearn.svm import SVC
#from sklearn.metrics import accuracy_score
#sns.set() #365 link