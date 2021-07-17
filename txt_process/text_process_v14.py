import warnings
warnings.filterwarnings("ignore")
#import pdfplumber
import re # add regular expression to take care of special characters and hyphens?
import sqlite3 
from matplotlib.backends.backend_pdf import PdfPages
import time
import os
import csv
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.ion() # take away popups - putting plot images in pdf

#from nltk import FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
#import nltk

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer # lsi
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
#from sklearn.cluster import AgglomerativeClustering
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import feature_extraction # br
from sklearn.cluster import MiniBatchKMeans
#from sklearn.decomposition import TruncatedSVD

#from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import Sparse2Corpus

from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

from sklearn.cluster import DBSCAN

#from scipy.cluster.hierarchy import dendrogram
#from scipy.cluster import hierarchy
import random
import seaborn as sns 
#import pyLDAvis
#import pyLDAvis.sklearn
#pyLDAvis.enable_notebook()
#365
sns.set() 
#365
paper_count = 3769874 

#vocab = dict()
def get_model_topics(model, vectorizer, topics, n_top_words=15): # towardsdatascience article 
    word_dict = {}
    feature_names = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        #word_dict[topics[topic_idx]] = top_features
        word_dict[topic_idx] = top_features

    return pd.DataFrame(word_dict)

def get_inference(model, vectorizer, topics, text, threshold): # towards data science article
    v_text = vectorizer.transform([text])
    score = model.transform(v_text)

    labels = set()
    for i in range(len(score[0])):
        if score[0][i] > threshold:
            labels.add(topics[i])

    if not labels:
        return 'None', -1, set()

    return topics[np.argmax(score)], score, labels

#def process_entries(path, output_dir, test_pred_dir, figure_dir):

def process_entries(path, results_dir, size):
    print("in here")
    global paper_count
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    #cur.execute("SELECT count(*) FROM astro_papers_2;")
    #paper_count = cur.fetchone()[0]
    #print(type(cur.fetchone()[0]))
    #conn.close()
    #print("connected")
    #run = 0 
    #while True:
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
        #df = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers;", conn)
        #print("got different frames")
        #df = pd.concat([df_1, df_2, df_3, df_4], axis=0)
        #kmeans_cluster(df, results_dir)
        #df['bibcode'] = df['bibcode'].astype(str)
        #df['title'] = df['title'].astype(str)
        #df['author'] = df['author'].astype(str)
        #df['abstract'] = df['abstract'].astype(str)
        #print(df.columns)
        #print("----")
        #print(df.dtypes)
        #df.to_csv('/home/maleeha/research/data.csv', index=False)
        #print(df.info(verbose=False, memory_usage="deep"))
        clustering(df, results_dir)
    elif size == 2:
        df_1 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 1980 AND year <1990 LIMIT 500;", conn)
        df_2 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 1990 AND year <2000 LIMIT 500;", conn)
        df_3 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 2000 AND year <2010 LIMIT 500;", conn)
        df_4 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 2010 AND year <2022 LIMIT 500;", conn)
        df = pd.concat([df_1, df_2, df_3, df_4], axis=0)
        print("DF INFO --------->")
        print(df.info(verbose=False, memory_usage="deep"))
        print()
        clustering(df, results_dir)
 
    conn.close()
    #return df

def parse_text(text):
    return re.findall("[a-zA-Z0-9']{2,}", text)

def tokenize(text):
    return [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]

def trunc(number): #so
    text = f"{number:.3f}"
    return float(text)

def stats(df):
    # print("Stats:")
    # r =  df['citation_count'].max() - df['citation_count'].min()
    # print("\trange of citation count:", r)
    stats_dict = {}

    Qmin = np.min(df['citation_count'])
    Q1 = np.percentile(df['citation_count'], 25, interpolation = 'midpoint') 
    Q3 = np.percentile(df['citation_count'], 75, interpolation = 'midpoint') 
    Q2 = np.median(df['citation_count'])
    Qmax = np.max(df['citation_count'])

    stats_dict.update({"Qmin": Qmin})
    stats_dict.update({"Q1":Q1})
    stats_dict.update({"Q3":Q3})
    stats_dict.update({"Q3":Q3})
    stats_dict.update({"Qmax":Qmax})
    
    # print("\tmin: ", Qmin)
    # print("\tQ1: ", Q1)
    # print("\tQ2: ", Q2)
    # print("\tQ3: ", Q3)
    # print("\tmax: ", Qmax)
    # print()
    # try_size(df)
    # a_file = open(results_dir + "stats.csv", "w")
    # stats = {'range':r, "min":Qmin, "q1":Q1, "q2":Q2, "q3":Q3, "max":Qmax}
    # writer = csv.writer(a_file) #kite
    # for key, value in stats.items():
    #     writer.writerow([key, value])
    # a_file.close()
    return stats_dict

def get_data_set(vect_no, data):
    if vect_no == 0:
        vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
        return count_vect.fit_transform(data)
    elif vect_no == 1:
        count_vect = CountVectorizer(stop_words='english',strip_accents='unicode',token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
        return count_vect.fit_transform(data)

    #terms = vect.get_feature_names()

# no == 0 --> lda, no == 1 --> lsi, 
def topic_model(tm_no, data):
    if tm_no == 0:
        svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
        svd.fit(data)
        data2D = svd.transform(data)
        return data2D

    elif tm_no == 1:
        lda = LatentDirichletAllocation(n_components=2, random_state=1)
        lda.fit(data)
        data_LDA = lda.transform(data)
        return data_LDA
    #X_tr = X_tr.todense()
            #pca = PCA(n_components=3).fit(X_tr)
            #d = pd.DataFrame(X_tr_c.toarray(),index=train['abstract'],columns=count_vect.get_feature_names()) # assn pdf
            #print(d.head())
            #X_tr_c = X_tr_c.todense()
            #print(X_tr_c)
            #print(len(list(count_vect.get_feature_names())))
            #print(len(count_vect.vocabulary))
            #corpus_vect = Sparse2Corpus(X_tr_c, documents_columns=False)
            #print(count_vect.vocabulary)
            #lsa_model = LsiModel(corpus_vect, id2word=count_vect.vocabulary)
            #print(lsa_model.print_topics())

def see_clusters(model, data):
    visualizer = SilhouetteVisualizer(model,  colors='yellowbrick')

    visualizer.fit(data)        # Fit the data to the visualizer
    visualizer.show() 

# model_no == 0 --> kmeans, model_no == 1 --> dbscan
def clustering(df, results_dir, model_no):    
    train, test = train_test_split(df, train_size=0.75)

    c_nums = [3, 6, 8, 10, 15, 25]
    
    print()
    with PdfPages(results_dir + "figures.pdf") as pdf: 
        for c in c_nums:
            print("Cluster={}".format(c))
            print()
            folder_str = ""
            model = None

            X_tr = get_data_set(0, train['abstract'])
            X_tr_c = get_data_set(1, train['abstract'])
            
            if model_no == 0: #kmeans
                folder_str = "./kmeans/"
                model_p = folder_str + "model_" + str(c) + ".pkl"
                model_ori_p = folder_str + "model_ori_" + str(c) + ".pkl"
                model = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=2000)
                model_ori = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=2000)
                data2D = topic_model(0, X_tr)

                model_ori.fit(X_tr)
                see_cluster(model_ori, X_tr)
                pickle.dump(model_ori, open(model_ori_p,'wb')) # machine learning mastery

                model.fit(data2D)
                see_cluster(model, data2D)
                pickle.dump(model, open(model_p, 'wb'))

            elif model_no == 1: #dbscan
                folder_str = "./dbs/"
                model = DBSCAN(eps=3, min_samples=2)
           

            new_train = pd.DataFrame(columns=train.columns)
        

        
            new_train = pd.concat([train.reset_index(drop=True), pd.DataFrame(data2D)], axis=1)
         
            new_train['cluster'] = model.labels_
            
            X_te = vect.transform(test['abstract'])
            #print("Xtr.shape",X_tr.shape)
            #print("train.shape", train.shape)
            #print("Xte.shape",X_te.shape)
            #print("test.shape", test.shape)
            
            test_pred = model_ori.predict(X_te)
            terms = vect.get_feature_names()

            silhouette_avg = silhouette_score(X_te, test_pred) #scikitlearn silhouette
            print("For n_clusters =", c, 
                    "The average silhouette_score is :", silhouette_avg)

            #print("Prediction scores on test data ", test_pred)
            #print(test.loc[0])
            test_dict = {}
            for i in range(X_te.shape[0]):
                #print("updating test_dict ...at value {x} out of {y}".format(x=i, y=len(X_te)))
                curr = test.iloc[i]
                #print(curr)
                #break
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

            c_file = open(results_dir + "c_terms" + str(c) + ".csv", "w")
            #stats = {'range':r, "min":Qmin, "q1":Q1, "q2":Q2, "q3":Q3, "max":Qmax}
            writer = csv.writer(c_file) #kite
            for key, value in cluster_terms_ori.items():
                writer.writerow([key, value])
            c_file.close()

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
            #print(set(arr))
            
            #print("Cluster Prediction of Test Set Plus Expected Citation Count")
            #print()
            #print(test_pred_df)
            #print()
            test_file = results_dir + "test_pred" + str(c) + ".csv"
            #test_file = results_dir + "test_pred" + str(c) + "run_{r}".format(r=run) + ".csv"
            test_pred_df.to_csv(test_file)

            cluster_colors = dict()
            for l in model.labels_:
                cluster_colors[l] = (random.random(), random.random(), random.random())
            #print("creating plot 1 ....")
            #print(centers2D)
           
            plot_clusters(pdf, c, new_train, 0, cluster_colors, centers2D, cluster_terms_ori)
            plot_clusters(pdf, c, new_train, 1, cluster_colors, centers2D, cluster_terms_ori)
            plot_clusters(pdf, c, new_train, 2, cluster_colors, centers2D, cluster_terms_ori)

def plot_clusters(pdf, c, new_train, ax_val, cluster_colors, centers2D, cluster_terms_ori):
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

def plot_results(path):
    #path = '/Users/Mal/Documents/results'
    with PdfPages("results.pdf") as pdf: 
        for filename in os.listdir(path):
            if "test_pred" in filename:
                name = os.path.join(path, filename)
                f = plt.figure()
                ax = sns.histplot(data=pd.read_csv(name), x='year', hue='pred_str')
                ax.set_title("File: " + name + " test prediction results")
                plt.show()
                pdf.savefig(f)
    
def main():
    #results_dir = "/home/maleeha/research/results/"
    results_dir = "/Users/Mal/Desktop/results/"
    
    #results_dir = "/Users/Mal/Documents/results/"

    if(os.path.exists(results_dir)==0):
        os.mkdir(results_dir)
    #path = '/Users/Mal/Desktop/sqlite/research.db'
    path = '/Users/Mal/Documents/research.db'
    #path = '/home/maleeha/research/research.db'
    #process_entries(path, results_dir, 0)
    process_entries(path, results_dir, 2)
    #path = '/Users/Mal/Desktop/resu'
    plot_results(results_dir)
    #process_entries(path, results_dir, 1)
    
    # import sys
    # from resource import getrusage, RUSAGE_SELF
    # process_entries(path, results_dir, 2)
    # print("Peak memory (MiB):",
    #   int(getrusage(RUSAGE_SELF).ru_maxrss / 1024))
    # #df = process_text(df)
    # print(np.polyfit([100, 1000], [1558488, 3593720], 1))
    
    #print(2.26e-03 * (3769874/ 1024) + 1.33e+06)
    #kmeans_cluster(df, output_dir_, test_pred_dir, figure_dir)
    #t0 = time.time()
    #plot_words(df) 
    #print("time elapsed for plot words ", time.time()-t0)
    #agg_cluster(df)
   
if __name__ == '__main__':
    main()

#-------------------------BELOW FROM CLUSTERING FUNCITON---------------------------------
   #        new_train = pd.concat([new_train.reset_index(drop=True), mini_df]) 
                    #model_ori = mini_kmeans(vect, mini_df, model_ori, 1)
            #while i <= len(train['abstract']):
            #    try:
            #        mini_df = train['abstract'][i:i+2000]
            #        X_tr = vect.fit_transform(mini_df)
            #        X_tr = X_tr.todense()
            #        pca = PCA(n_components=3).fit(X_tr)
            #        data2D = pca.transform(X_tr)
                    #model = mini_kmeans(vect, mini_df, model, 0)
            #        print("not data2d", i, X_tr.shape)
            #        print(i, data2D.shape)
            #        model = model.partial_fit(data2D)
            #        model_ori = model_ori.partial_fit(X_tr)
            #        mini_df = pd.concat([mini_df.reset_index(drop=True), pd.DataFrame(data2D)], axis=1)
            #        new_train = pd.concat([new_train.reset_index(drop=True), mini_df]) 
                    #model_ori = mini_kmeans(vect, mini_df, model_ori, 1)
            #        i += 2000
            #    except IndexError:
            #        break
            #df_new = pd.concat([train.reset_index(drop=True), pd.DataFrame(data2D)], axis=1)
            #df_new = pd.concat([train.reset_index(drop=True), pd.DataFrame(data2D)], axis=1) 
            #print(new_train.columns) 
            #print(new_train.head())  
            #print("new_train.shape", new_train.shape)

              
    # i = 0
    # count = 0
    # while i != train.shape[0]:
    #     X_tr = vect.fit_transform(train.iloc[i:i+50]['abstract'])
    #     print("run ... ",count )
    #     print(vect.get_feature_names()[:50])
    #     count+=1
    #     i+=50

    #print(vect.get_feature_names()[:25])

    #d = pd.DataFrame(lsa.components_,index = ["component_1","component_2"],columns = count_vect.get_feature_names())
            #print(d.head())
            
            #pyLDAvis.sklearn.prepare(lda, X_tr_c, count_vect)

            #print(get_model_topics(lda_tfidf, vect, []))
            #print(get_model_topics(lda, count_vect, []))

            #topic, score, _ = get_inference(nmf, tfidf_vectorizer, nmf_topics, text, 0)
            #print(topic, score)
            #print(train.columns)
            #for x in train.columns:
            #    print(x, type(x))  


# def try_size(df):
#     vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
#     train, test = train_test_split(df, train_size=0.90)
#     X_te = vect.fit_transform(test['abstract'])

# def mini_kmeans(vect, df, model, data):
#     X_tr = vect.fit_transform(df)
#     X_tr = X_tr.todense()
#     if data == 0:
#         pca = PCA(n_components=3).fit(X_tr)
#         data2D = pca.transform(X_tr)
#         model = model.partial_fit(data2D)
#     elif data == 1:
#         model = model.partial_fit(X_tr)
#     return model

'''
f = plt.figure()
x_axis = new_train['Component 2']
y_axis = new_train['Component 1']
#plt.figure(figsize=(10,8))
ax = sns.scatterplot(x_axis, y_axis, hue = new_train['cluster'], palette=list(cluster_colors.values()))
ax.legend(fontsize=6)
xs = [p[1] for p in centers2D] # link
ys = [p[0] for p in centers2D]
#print(xs, ys)
#print(cluster_terms_ori.items())
if c == 3:
    for i, txt in enumerate(cluster_terms_ori.items()): # link
        print(txt[1], type(txt[1]))
        text = "    " + str(i) + " " + txt[1]
        plt.annotate(text, (xs[i], ys[i]), (xs[i], ys[i]),fontsize='medium',c='k', weight='bold') #link
    plt.scatter(centers2D[:,1], centers2D[:,0], marker='*', s=125, linewidths=3, c='k')
plt.title("{x}-means Clusters - PCA components 2 v. 1".format(x=c))
plt.show()
#f.savefig("component2v1.pdf", bbox_inches='tight')
pdf.savefig(f)

#print("creating plot 2 ....")
#print(centers2D)
f2 = plt.figure()
x_axis = new_train['Component 3']
y_axis = new_train['Component 2']
#plt.figure(figsize=(10,8))
ax = sns.scatterplot(x_axis, y_axis, hue = new_train['cluster'], palette=list(cluster_colors.values()))
ax.legend(fontsize=6)
xs = [p[2] for p in centers2D] # link
ys = [p[1] for p in centers2D]
#print(xs, ys)
if c == 3:
    for i, txt in enumerate(cluster_terms_ori.items()): # link
        text = "    " + str(i) + " " + txt[1]
        plt.annotate(text, (xs[i], ys[i]), (xs[i], ys[i]),fontsize='medium',c='k', weight='bold') #link
    plt.scatter(centers2D[:,2], centers2D[:,1], marker='*', s=125, linewidths=3, c='k')
plt.title("{x}-means Clusters - PCA components 3 v. 2".format(x=c))
plt.show()
pdf.savefig(f2)
#f.savefig("component3v2.pdf", bbox_inches='tight')
#print("creating plot 3 .....")
f3 = plt.figure()
x_axis = new_train['Component 3']
y_axis = new_train['Component 1']
#plt.figure(figsize=(10,8))
ax = sns.scatterplot(x_axis, y_axis, hue = new_train['cluster'], palette=list(cluster_colors.values()))
ax.legend(fontsize=6)
xs = [p[2] for p in centers2D] # link
ys = [p[0] for p in centers2D]
#print(xs, ys)
if c == 3:
    for i, txt in enumerate(cluster_terms_ori.items()): # link
        text = "    " + str(i) + " " + txt[1]
        plt.annotate(text, (xs[i], ys[i]), (xs[i], ys[i]),fontsize='medium',c='k', weight='bold') # link
    plt.scatter(centers2D[:,2], centers2D[:,0], marker='*', s=125, linewidths=3, c='k')
plt.title("{x}-means Clusters - PCA components 3 v. 1".format(x=c))
plt.show()
pdf.savefig(f3)
#f.savefig("component3v1.pdf", bbox_inches='tight')
#print("done making plots ....")


run = 0
cmd = cur.execute("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2;")
#df_count = 0
run = 0
while True:
results = cmd.fetchmany(size=10000)
if len(results) == 0:
break
print("Paper count left ... {x}, current run: {r}".format(x=paper_count, r=run))
#cmd = cur.execute("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 LIMIT 200;")
df = pd.DataFrame(results)
df.columns = list(map(lambda x: x[0], cur.description)) #so
#df = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 LIMIT 200;", conn)
t0 = time.time()
kmeans_cluster(df, run, output_dir, test_pred_dir, figure_dir)
print("time taken for 10000 ", time.time() - t0)
paper_count -= 10000
run +=1
'''