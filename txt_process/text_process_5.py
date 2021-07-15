#import pdfplumber
import re # add regular expression to take care of special characters and hyphens?
import sqlite3 
from matplotlib.backends.backend_pdf import PdfPages
import time

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

#vocab = dict()

def read_entries(path):
    conn = sqlite3.connect(path)
    df = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 LIMIT 100;", conn)
    conn.close()
    return df

def parse_text(text):
    return re.findall("[a-zA-Z0-9']{2,}", text)

def tokenize(text):
    return [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]

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
    vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))

    X = vect.fit_transform(df['abstract'])
    vocab = vect.vocabulary_
    print("vocab len ", len(vocab))
    years = []
    for i in range(1980, 2022):
        years.append(i)

    counts = pd.DataFrame(index=vocab, columns=years)

    for token in vocab:
        arr = []
        for i in range(1980, 2022):
            d = df[df['year'] == i]
            #print(d)
            year_count = 0
            for index, row in d.iterrows():
                year_count += word_count(token, row['abstract'])
            arr.append(year_count)
        counts.loc[token] = arr
    print(counts.head())

    counts = counts.fillna(0) #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
    #print(counts.info)
    
    counts.append(counts.sum(numeric_only=True, axis=1), ignore_index=True)
    print(counts.head())

    for col in counts.columns:
        print(col, counts[col].nlargest())

    # count = 0
    # for index, row in counts.iterrows():
    #     if count == 10:
    #         break
    #     print(index, row)
    #     count +=1

    #print(counts)

    #counts.to_csv('/Users/Mal/Desktop/research/data.csv')

def trunc(number): #so
    text = f"{number:.3f}"
    return float(text)

def kmeans_cluster(df):
    #link
    
    vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))

    X = vect.fit_transform(df['abstract']) #.iloc[:80]
    
    #vocab = vect.vocabulary_
    #idf = vect.idf_
    #print(vect.get_feature_names())
    #print(X.shape)
    X_dense = X.todense() # link
    #print(X_dense.shape)
    X_train = X_dense[:80]
    X_test = X_dense[80:]
    print(X_train.shape)
    print(X_test.shape)
    # ncomponents = 3
    pca = PCA(n_components=3).fit(X_dense)
    data2D = pca.transform(X_dense)

    terms = vect.get_feature_names()
    c_nums = [3, 6, 8, 10, 15, 25] # changed 50 to 40 bc of convergence warnings
    #c_nums = [3]
    print("K-Means Clustering =====>")
    print()
    
    with PdfPages('figures.pdf') as pdf:
        for c in c_nums:
            print("Cluster={}".format(c))
            model = KMeans(n_clusters=c)
            model.fit(data2D)
            #print("model labels ", model.labels_, model.labels_.shape)
        
            df_new = pd.concat([df.reset_index(drop=True), pd.DataFrame(data2D)], axis=1)
            df_new['cluster'] = model.labels_
                
            model_ori = KMeans(n_clusters=c)
            model_ori.fit(X_dense)

            centers2D = model.cluster_centers_
            centers2D_ori = model_ori.cluster_centers_
            #print("centers 2d", centers2D, centers2D.shape)
            #print("centers 2d ori", centers2D_ori, centers2D_ori.shape)
            order_centroids = centers2D.argsort()[:, ::-1]  
            order_centroids_ori = centers2D_ori.argsort()[:, ::-1]  
            #print("oc " , order_centroids)
            #print("oc ori ", order_centroids_ori)
            cluster_terms = []
            cluster_terms_ori = []
            for i in range(c):
                c_terms = " ".join([terms[ind] for ind in order_centroids[i, :c]])
                cluster_terms.append(c_terms)
                c_terms_ori = " ".join([terms[ind] for ind in order_centroids_ori[i, :c]])
                cluster_terms_ori.append(c_terms_ori)

            #print("Cluster Terms ", cluster_terms_ori)
            #print("cluster terms ori ", cluster_terms_ori)
            
            df_new.rename(columns={0: 'Component 1', 1 : 'Component 2', 2: 'Component 3'}, inplace=True)
            means = df_new.groupby(['cluster']).mean()
            means['citation_count'] = means['citation_count'].apply(trunc)
            print("means", means)
            cluster_colors = dict()
            for l in model.labels_:
                cluster_colors[l] = (random.random(), random.random(), random.random())

            #print(centers2D)
            f = plt.figure()
            x_axis = df_new['Component 2']
            y_axis = df_new['Component 1']
            #plt.figure(figsize=(10,8))
            ax = sns.scatterplot(x_axis, y_axis, hue = df_new['cluster'], palette=list(cluster_colors.values()))
            ax.legend(fontsize=6)
            xs = [p[1] for p in centers2D] # link
            ys = [p[0] for p in centers2D]
            #print(xs, ys)
            if c == 3:
                for i, txt in enumerate(cluster_terms_ori): # link
                    text = "    " + str(i) + " " + txt
                    plt.annotate(text, (xs[i], ys[i]), (xs[i], ys[i]),fontsize='medium',c='k', weight='bold') #link
                plt.scatter(centers2D[:,1], centers2D[:,0], marker='*', s=125, linewidths=3, c='k')
            plt.title("{x}-means Clusters - PCA components 2 v. 1".format(x=c))
            plt.show()
            #f.savefig("component2v1.pdf", bbox_inches='tight')
            pdf.savefig(f)

            f2 = plt.figure()
            x_axis = df_new['Component 3']
            y_axis = df_new['Component 2']
            #plt.figure(figsize=(10,8))
            ax = sns.scatterplot(x_axis, y_axis, hue = df_new['cluster'], palette=list(cluster_colors.values()))
            ax.legend(fontsize=6)
            xs = [p[2] for p in centers2D] # link
            ys = [p[1] for p in centers2D]
            #print(xs, ys)
            if c == 3:
                for i, txt in enumerate(cluster_terms_ori): # link
                    text = "    " + str(i) + " " + txt
                    plt.annotate(text, (xs[i], ys[i]), (xs[i], ys[i]),fontsize='medium',c='k', weight='bold') #link
                plt.scatter(centers2D[:,2], centers2D[:,1], marker='*', s=125, linewidths=3, c='k')
            plt.title("{x}-means Clusters - PCA components 3 v. 2".format(x=c))
            plt.show()
            pdf.savefig(f2)
            #f.savefig("component3v2.pdf", bbox_inches='tight')

            f3 = plt.figure()
            x_axis = df_new['Component 3']
            y_axis = df_new['Component 1']
            #plt.figure(figsize=(10,8))
            ax = sns.scatterplot(x_axis, y_axis, hue = df_new['cluster'], palette=list(cluster_colors.values()))
            ax.legend(fontsize=6)
            xs = [p[2] for p in centers2D] # link
            ys = [p[0] for p in centers2D]
            #print(xs, ys)
            if c == 3:
                for i, txt in enumerate(cluster_terms_ori): # link
                    text = "    " + str(i) + " " + txt
                    plt.annotate(text, (xs[i], ys[i]), (xs[i], ys[i]),fontsize='medium',c='k', weight='bold') # link
                plt.scatter(centers2D[:,2], centers2D[:,0], marker='*', s=125, linewidths=3, c='k')
            plt.title("{x}-means Clusters - PCA components 3 v. 1".format(x=c))
            plt.show()
            pdf.savefig(f3)
            #f.savefig("component3v1.pdf", bbox_inches='tight')
            
def main():
    path = '/Users/Mal/Desktop/sqlite/research.db'
    df = read_entries(path)
    #df = process_text(df)
    kmeans_cluster(df)
    #t0 = time.time()
    #plot_words(df) 
    #print("time elapsed for plot words ", time.time()-t0)
    #agg_cluster(df)
   
if __name__ == '__main__':
    main()
