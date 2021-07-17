#import pdfplumber
import re # add regular expression to take care of special characters and hyphens?
import sqlite3 
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion() # take away popups - putting plot images in pdf

#from nltk import FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
#import nltk

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
#from sklearn.cluster import AgglomerativeClustering
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import feature_extraction # br

#from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from collections import defaultdict
from sklearn.decomposition import PCA

#from scipy.cluster.hierarchy import dendrogram
#from scipy.cluster import hierarchy
import random
import seaborn as sns #365
sns.set() #365


#from scipy.cluster.vq import vq, kmeans  #sh github

vocab = defaultdict(int)
vocab_tokens = defaultdict(int)
vocab_stems = defaultdict(int)
vocab_bigrams = defaultdict(int)
#vocab_trigrams = defaultdict(int)

def parse_text(text):
    return re.findall("([^(.*)][a-zA-Z0-9']{2,})", text)

def read_entries(path):
    conn = sqlite3.connect(path)
    df = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 LIMIT 250;", conn)
    conn.close()
    return df

def tokenize(text): #stack overflow
    return [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]

def tokenize_bigram(text): #stack overflow
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    return list(ngrams(tokens, 2))

def tokenize_trigram(text): #stack overflow
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    return list(ngrams(tokens, 3))

def tokenize_and_stem(text): #br
    stemmer = SnowballStemmer('english')
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems  

def process_text(df):
    #stop_words = stopwords.words('english')
    abs_text = df['abstract']
    abs_text = abs_text.str.lower() 

    abs_text = abs_text.apply(parse_text)
    abs_text = abs_text.apply(' '.join)

    abs_text_stem = abs_text.apply(tokenize_and_stem)
    abs_text_token = abs_text.apply(tokenize)
    abs_text_bigram = abs_text.apply(tokenize_bigram)
  
    df['abs_text'] = abs_text_token
    
    tokenized = []
    for abs_text in df['abs_text']:
        abstract_str = " ".join(abs_text)
        tokenized.append(abstract_str)

    df['abs_tokenized'] = tokenized  # preprocessing abstract text, tokenizing then rejoining

    for abs_text in abs_text_token:
        for term in abs_text:
            vocab_tokens[term] +=1 

    for abs_text in abs_text_stem:
        for term in abs_text:
            vocab_stems[term] +=1 

    for abs_text in abs_text_bigram:
        for term in abs_text:
            vocab_bigrams[term] += 1 #creating token, stem, bigram dictionaries

    return df

def trunc(number):  #stack overflow
    text = f"{number:.3f}"
    return float(text)

def word_count(word, text):

    text = text.lower() 
    text = parse_text(text)
    text = ' '.join(text)
    text_tokens = tokenize(text)

    count = 0
    for tk in text_tokens:
        if word == tk:
            count+=1
    return count

def plot_words(df): # data analysis 
    years = []
    for i in range(1980, 2022):
        years.append(i)

    counts = pd.DataFrame(index=vocab_tokens, columns=years)
    for token in vocab_tokens:
        arr = []
        for i in range(1980, 2022):
            d = df[df['year'] == i]
            #print(d)
            year_count = 0
            for index, row in d.iterrows():
                year_count += word_count(token, row['abs_tokenized'])
            arr.append(year_count)
        counts.loc[token] = arr
    print(counts)

    counts = counts.fillna(0) #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
    print(counts.info)
    for col in counts.columns:
        counts[col] = counts[col].sort_values(ascending=False)

    count = 0
    for index, row in counts.iterrows():
        if count == 10:
            break
        print(index, row)
        count +=1

    #counts.to_csv('/Users/Mal/Desktop/research/data.csv')
 
def label_point(x, y, val, ax): #https://stackoverflow.com/questions/46027653/adding-labels-in-x-y-scatter-plot-with-seaborn
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))

def kmeans_cluster(df):
    #link
    vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))

    X = vect.fit_transform(df['abstract'])
    vocab = vect.vocabulary_
    idf = vect.idf_
    print(vect.get_feature_names())
    X_dense = X.todense() # link

    # ncomponents = 3
    pca = PCA(n_components=3).fit(X_dense)
    data2D = pca.transform(X_dense)

    terms = vect.get_feature_names()
    c_nums = [3, 6, 8, 10, 15, 25] # changed 50 to 40 bc of convergence warnings
    print("K-Means Clustering =====>")
    print()

    with PdfPages('figures.pdf') as pdf:
        for c in c_nums:
            print("Cluster={}".format(c))
            model = KMeans(n_clusters=c, init='k-means++', max_iter=200, n_init=10)
            model.fit(data2D)

            df_new = pd.concat([df.reset_index(drop=True), pd.DataFrame(data2D)], axis=1)
            df_new['cluster'] = model.labels_
                
            model_ori = KMeans(n_clusters=c, init='k-means++', max_iter=200, n_init=10)
            model_ori.fit(X_dense)

            order_centroids = model.cluster_centers_.argsort()[:, ::-1]  
            cluster_terms = []

            for i in range(c):
                c_terms = " ".join([terms[ind] for ind in order_centroids[i, :c]])
                cluster_terms.append(c_terms)
        
            term_dict = {}
            count = 0
            for val in df_new['cluster'].unique():
                term_dict[val] = cluster_terms[count]
                count+=1

            #print(term_dict)
            terms = []
            for val in df_new['cluster']:
                terms.append(term_dict[val])

            df_new['terms'] = terms
            #print(terms)
            df_new.rename(columns={0: 'Component 1', 1 : 'Component 2', 2: 'Component 3'}, inplace=True)

            cluster_colors = dict()
            for l in model.labels_:
                cluster_colors[l] = (random.random(), random.random(), random.random())

            centers2D = model.cluster_centers_
            x_axis = [p[0] for p in centers2D] # link
            y_axis = [p[1] for p in centers2D]
            
            f = plt.figure()
            x_axis = df_new['Component 2']
            y_axis = df_new['Component 1']
            #plt.figure(figsize=(10,8))
            ax = sns.scatterplot(x_axis, y_axis, hue = df_new['cluster'], palette=list(cluster_colors.values()))
            plt.title("Cluster by PCA components 2 v. 1")
            plt.show()
            #f.savefig("component2v1.pdf", bbox_inches='tight')
            pdf.savefig(f)

            x_axis = df_new['Component 3']
            y_axis = df_new['Component 2']
            #plt.figure(figsize=(10,8))
            ax = sns.scatterplot(x_axis, y_axis, hue = df_new['cluster'], palette=list(cluster_colors.values()))
            plt.title("Cluster by PCA components 3 v. 2") 
            plt.show()
            pdf.savefig(f)
            #f.savefig("component3v2.pdf", bbox_inches='tight')

            x_axis = df_new['Component 3']
            y_axis = df_new['Component 1']
            #plt.figure(figsize=(10,8))
            ax = sns.scatterplot(x_axis, y_axis, hue = df_new['cluster'], palette=list(cluster_colors.values()))
            plt.title("Cluster by PCA components 3 v. 1")
            plt.show()
            pdf.savefig(f)
            #f.savefig("component3v1.pdf", bbox_inches='tight')
            
            #break
    
def main():
    path = '/Users/Mal/Desktop/sqlite/research.db'
    df = read_entries(path)
    df = process_text(df)
    #plot_words(df)
    kmeans_cluster(df)
    #agg_cluster(df)
   
if __name__ == '__main__':
    main()
