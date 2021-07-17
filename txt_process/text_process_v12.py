import warnings
warnings.filterwarnings("ignore")
#import pdfplumber
import sqlite3 
from matplotlib.backends.backend_pdf import PdfPages
import time
import os
import csv

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
from sklearn.cluster import MiniBatchKMeans
import dask_ml.feature_extraction.text


#from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split

from collections import defaultdict
from sklearn.decomposition import PCA

#from scipy.cluster.hierarchy import dendrogram
#from scipy.cluster import hierarchy
import random
import seaborn as sns 
#365
sns.set() 
#365
paper_count = 3769874 

def process_entries(path, results_dir, size):
    print("in here")
    global paper_count
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    if size == 0:
        print("am doing")
        df_1 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 1980 AND year <1990 LIMIT 25;", conn)
        df_2 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 1990 AND year <2000 LIMIT 25;", conn)
        df_3 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 2000 AND year <2010 LIMIT 25;", conn)
        df_4 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 2010 AND year <2022 LIMIT 25;", conn)
        df = pd.concat([df_1, df_2, df_3, df_4], axis=0)
        clustering(df, results_dir)
    elif size == 1:
        df = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2;", conn)
        df.to_csv('/home/maleeha/research/data.csv', index=False)
        #print(df.info(verbose=False, memory_usage="deep"))
        #clustering(df, results_dir)
    conn.close()
    #return df

def trunc(number): #so
    text = f"{number:.3f}"
    return float(text)

import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
def process_csv(path, results_dir):
    print("hello")
    t0 = time.time()
    df = dd.read_csv(path)

    print("Hello time ", time.time() -t0)
    X = df['abstract'].to_dask_array(lengths=True)
    #print(df.to_dask_array())
    print(X)
    
    vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
    print(df.head())
    print(type(df))
    vect = dask_ml.feature_extraction.text.HashingVectorizer()
    
    
    train, test = train_test_split(X, train_size=0.90)

    X_tr = vect.fit_transform(train)
    pca = PCA(n_components=3).fit(X_tr)
    data2D = pca.transform(X_tr)
    new_train = pd.DataFrame(columns=df.columns)
    km = dask_ml.cluster.KMeans(n_clusters=3, init_max_iter=2, oversampling_factor=10)
    km.fit(X)
    #model = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=2000)
    #model_ori = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=2000)
    
    #model.fit(data2D)
    #model_ori.fit(X_tr)
    
    new_train = pd.concat([train.reset_index(drop=True), pd.DataFrame(data2D)], axis=1)
    new_train['cluster'] = km.labels_
    
    X_te = vect.transform(test)
    
    test_pred = model_ori.predict(X_te)
    terms = vect.get_feature_names()


def clustering(df, results_dir):
    # getting stats of citation_count and writing to file
    r =  df['citation_count'].max() - df['citation_count'].min()

    Qmin = np.min(df['citation_count'])
    Q1 = np.percentile(df['citation_count'], 25, interpolation = 'midpoint') 
    Q3 = np.percentile(df['citation_count'], 75, interpolation = 'midpoint') 
    Q2 = np.median(df['citation_count'])
    Qmax = np.max(df['citation_count'])
    
    print("Stats: -> citation count")
    print("\trange:", r)
    print("\tmin: ", Qmin)
    print("\tQ1: ", Q1)
    print("\tQ2: ", Q2)
    print("\tQ3: ", Q3)
    print("\tmax: ", Qmax)
    print()

    a_file = open(results_dir + "stats.csv", "w")
    stats = {'range':r, "min":Qmin, "q1":Q1, "q2":Q2, "q3":Q3, "max":Qmax}
    writer = csv.writer(a_file) #kite
    for key, value in stats.items():
        writer.writerow([key, value])
    a_file.close()

    # tf-idf matrix and train/test split
    print(df.head())
    vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
    train, test = train_test_split(df, train_size=0.90)
    
    c_nums = [3, 6, 8, 10, 15, 25]
    
    # i = 0
    # count = 0
    # while i != train.shape[0]:
    #     X_tr = vect.fit_transform(train.iloc[i:i+50]['abstract'])
    #     print("run ... ",count )
    #     print(vect.get_feature_names()[:50])
    #     count+=1
    #     i+=50

    # print(vect.get_feature_names()[:25])
    # print()

    with PdfPages(results_dir + "figures.pdf") as pdf: 
        for c in c_nums:
            print("Cluster={}".format(c))
            print()
            X_tr = vect.fit_transform(train['abstract'])
            X_tr = X_tr.todense()
            pca = PCA(n_components=3).fit(X_tr)
            data2D = pca.transform(X_tr)
            new_train = pd.DataFrame(columns=train.columns)
            model = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=2000)
            model_ori = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=2000)
            
            model.fit(data2D)
            model_ori.fit(X_tr)
            
            new_train = pd.concat([train.reset_index(drop=True), pd.DataFrame(data2D)], axis=1)
            new_train['cluster'] = model.labels_
            
            X_te = vect.transform(test['abstract'])
           
            test_pred = model_ori.predict(X_te)
            terms = vect.get_feature_names()

            test_dict = {}
            for i in range(X_te.shape[0]):
                curr = test.iloc[i]
                test_dict.update({tuple(curr.values):test_pred[i]})
                test_dict.update({(test.iloc[i]['bibcode'], test.iloc[i]['title']):test_pred[i]})
            #return
            #print("test dict ", test_dict)
            centers2D = model.cluster_centers_
            centers2D_ori = model_ori.cluster_centers_
            #print("centers 2d", centers2D, centers2D.shape)
            #print("centers 2d ori", centers2D_ori, centers2D_ori.shape)
            order_centroids = centers2D.argsort()[:, ::-1]  
            order_centroids_ori = centers2D_ori.argsort()[:, ::-1]  
            #print("oc " , order_centroids)
            #print("oc ori ", order_centroids_ori)
            cluster_terms = []
            #cluster_terms_ori = []
            cluster_terms_ori = {}
            for i in range(c):
                c_terms = " ".join([terms[ind] for ind in order_centroids[i, :c]])
                cluster_terms.append(c_terms)
                c_terms_ori = " ".join([terms[ind] for ind in order_centroids_ori[i, :c]])
                #cluster_terms_ori.append((i, c_terms_ori))
                cluster_terms_ori[i] = c_terms_ori

            #file = open(results_dir + "output.txt", "a")
            #file = open(output_dir + "output_run{r}.txt".format(r=run), "a")
            #print("Cluster Terms")
            #file = open(results_dir + "c_terms.csv", "a")

            c_file = open(results_dir + "c_terms" + str(c) + ".csv", "w")
            #stats = {'range':r, "min":Qmin, "q1":Q1, "q2":Q2, "q3":Q3, "max":Qmax}
            writer = csv.writer(c_file) #kite
            for key, value in cluster_terms_ori.items():
                writer.writerow([key, value])
            c_file.close()
            # file.write("\tCluster Terms\n")
            # for x in cluster_terms_ori:
            #     #print("\t" + str(x[0]) , x[1])
            #     file.write("\t" + str(x[0]) + "\t" + x[1] + "\n")
            # #print()
            # #print("cluster terms ori ", cluster_terms_ori)
            # file.close()

            new_train.rename(columns={0: 'Component 1', 1 : 'Component 2', 2: 'Component 3'}, inplace=True)
            #print("columns b4 means", new_train.columns)
            #print(new_train.groupby(['cluster']).head())
            #means = new_train.groupby(['cluster']).mean()
            #print("means columns", means.columns)
            #print(means.head())
            #print("new_train.columns", new_train.columns)
            #for x in new_train.columns:
            #    print(new_train[x].dtype)
            new_train = new_train.astype({'citation_count': 'float64'})
            means = new_train.groupby(['cluster'], as_index=False)
            #means = df_new.groupby(['cluster']).mean()
            #print(means.head())
            means = means.mean()
            #print("means.columns", means.columns)
            #print(means.head())

            means['citation_count'] = means['citation_count'].apply(trunc)
            #print("means", list(means.index))
            #means['cluster'] = means.index
            
            test_pred_df = pd.DataFrame(columns=list(df.columns) + [ 'predicted_cluster', 'expected_cit_count', 'pred_str'])
            #test_pred_df = pd.DataFrame(columns=['bibcode', 'title', 'predicted_cluster', 'expected_cit_count', 'pred_str'])
            #print(test_pred_df.columns)
            #break
            p_count = 0
            arr = []
            for x in test_dict.items():
                #print(x)
                #print("here")
                m = means.loc[x[1]]
                #print(x, m)
                #print("making test_pred dictionary...at paper {p}".format(p=p_count))
                pred_str = ""
                exp_cit_count = m['citation_count']
                if exp_cit_count < Q1:
                    pred_str = "low"
                elif exp_cit_count >= Q1 and exp_cit_count <= Q3:
                    pred_str = 'medium'
                elif exp_cit_count > Q3:
                    pred_str = "high"
                
                #for keys in x:
                #print(x[0])
                try:
                    test_pred_df = test_pred_df.append({'bibcode':x[0][0], 'title':x[0][1], \
                        'year':x[0][2], 'author':x[0][3], 'abstract':x[0][4], 'citation_count':x[0][5],'predicted_cluster':x[1],\
                     'expected_cit_count':exp_cit_count, 'pred_str':pred_str}, ignore_index=True)
                except IndexError:
                    #print(len(x))
                    arr.append(len(x))
                    test_pred_df = test_pred_df.append({'bibcode':x[0][0], 'title':x[0][1], \
                        'predicted_cluster':x[1],\
                     'expected_cit_count':exp_cit_count, 'pred_str':pred_str}, ignore_index=True)
                #print(x[0], "predicted cluster: ", x[1], "avg citation count of cluster {x}: ".format(x=x[1]), m['citation_count'])
                #print(m)
                p_count+=1
           
            cluster_colors = dict()
            for l in model.labels_:
                cluster_colors[l] = (random.random(), random.random(), random.random())
            
            test_file = results_dir + "test_pred" + str(c) + ".csv"
            test_pred_df.to_csv(test_file)

            plot_data(pdf, c, new_train, 0, cluster_colors, centers2D, cluster_terms_ori)
            plot_data(pdf, c, new_train, 1, cluster_colors, centers2D, cluster_terms_ori)
            plot_data(pdf, c, new_train, 2, cluster_colors, centers2D, cluster_terms_ori)

def plot_data(pdf, c, new_train, ax_val, cluster_colors, centers2D, cluster_terms_ori):
    f = plt.figure()
    #print(centers2D)
    if ax_val == 0:
        x_axis = new_train['Component 2']
        y_axis = new_train['Component 1']
        title = "PCA Components 2 v. 1"
    elif ax_val == 1:
        x_axis = new_train['Component 3']
        y_axis = new_train['Component 2']
        title = "PCA Components 3 v. 2"
    elif ax_val == 2:
        x_axis = new_train['Component 3']
        y_axis = new_train['Component 1']
        title = "PCA Components 3 v. 1"
    
    #plt.figure(figsize=(10,8))
    ax = sns.scatterplot(x_axis, y_axis, hue = new_train['cluster'], palette=list(cluster_colors.values()))
    ax.legend(fontsize=6)
    xs = [p[1] for p in centers2D] # link
    ys = [p[0] for p in centers2D]
    #print(xs, ys)
    #print(cluster_terms_ori.items())
    if c == 3:
        for i, txt in enumerate(cluster_terms_ori.items()): # link
            #print(txt[1], type(txt[1]))
            text = "    " + str(i) + " " + txt[1]
            plt.annotate(text, (xs[i], ys[i]), (xs[i], ys[i]),fontsize='medium',c='k', weight='bold') #link
        plt.scatter(centers2D[:,1], centers2D[:,0], marker='*', s=125, linewidths=3, c='k')
    plt.title(("{x}-means Clusters -" + title).format(x=c))
    plt.show()
    #f.savefig("component2v1.pdf", bbox_inches='tight')
    pdf.savefig(f)
               
def main():
    #results_dir = "/home/maleeha/research/results/"
    #results_dir = "/Users/Mal/Desktop/results/"
    results_dir = "/Users/Mal/Documents/results/"
    if(os.path.exists(results_dir)==0):
        os.mkdir(results_dir)
    #path = '/Users/Mal/Desktop/sqlite/research.db'
    path = '/Users/Mal/Documents/research.db'
    #path = '/home/maleeha/research/research.db'
    process_entries(path, results_dir, 0)
    #process_csv('/home/maleeha/research/data.csv', results_dir)
    '''
    import sys
    from resource import getrusage, RUSAGE_SELF
    process_entries(path, results_dir, 2)
    print("Peak memory (MiB):",
      int(getrusage(RUSAGE_SELF).ru_maxrss / 1024))
    #df = process_text(df)
    print(np.polyfit([100, 1000], [1558488, 3593720], 1))
    '''
    #print(2.26e-03 * (3769874/ 1024) + 1.33e+06)
    #kmeans_cluster(df, output_dir_, test_pred_dir, figure_dir)
    #t0 = time.time()
    #plot_words(df) 
    #print("time elapsed for plot words ", time.time()-t0)
    #agg_cluster(df)
   
if __name__ == '__main__':
    main()
