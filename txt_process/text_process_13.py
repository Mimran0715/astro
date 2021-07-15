import warnings
warnings.filterwarnings("ignore")
#import pdfplumber
import re # add regular expression to take care of special characters and hyphens?
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
#from sklearn.decomposition import TruncatedSVD

#from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

#vocab = dict()

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
        df_1 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 1980 AND year <1990 LIMIT 250;", conn)
        df_2 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 1990 AND year <2000 LIMIT 250;", conn)
        df_3 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 2000 AND year <2010 LIMIT 250;", conn)
        df_4 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 2010 AND year <2022 LIMIT 250;", conn)
        df = pd.concat([df_1, df_2, df_3, df_4], axis=0)
        print(df.info(verbose=False, memory_usage="deep"))
        clustering(df, results_dir)
    ''' 
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
    conn.close()
    #return df

def parse_text(text):
    return re.findall("[a-zA-Z0-9']{2,}", text)

def tokenize(text):
    return [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]

def trunc(number): #so
    text = f"{number:.3f}"
    return float(text)

def try_size(df):
    vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
    train, test = train_test_split(df, train_size=0.90)
    X_te = vect.fit_transform(test['abstract'])


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

def clustering(df, results_dir):
    print("Stats:")
    r =  df['citation_count'].max() - df['citation_count'].min()
    print("\trange of citation count:", r)

    Qmin = np.min(df['citation_count'])
    Q1 = np.percentile(df['citation_count'], 25, interpolation = 'midpoint') 
    Q3 = np.percentile(df['citation_count'], 75, interpolation = 'midpoint') 
    Q2 = np.median(df['citation_count'])
    Qmax = np.max(df['citation_count'])
    
    print("\tmin: ", Qmin)
    print("\tQ1: ", Q1)
    print("\tQ2: ", Q2)
    print("\tQ3: ", Q3)
    print("\tmax: ", Qmax)
    print()
    try_size(df)
    a_file = open(results_dir + "stats.csv", "w")
    stats = {'range':r, "min":Qmin, "q1":Q1, "q2":Q2, "q3":Q3, "max":Qmax}
    writer = csv.writer(a_file) #kite
    for key, value in stats.items():
        writer.writerow([key, value])
    a_file.close()

    vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
    train, test = train_test_split(df, train_size=0.90)
    
    #terms = vect.get_feature_names()
    c_nums = [3, 6, 8, 10, 15, 25]
    
    # i = 0
    # count = 0
    # while i != train.shape[0]:
    #     X_tr = vect.fit_transform(train.iloc[i:i+50]['abstract'])
    #     print("run ... ",count )
    #     print(vect.get_feature_names()[:50])
    #     count+=1
    #     i+=50

    #print(vect.get_feature_names()[:25])
    print()
    with PdfPages(results_dir + "figures.pdf") as pdf: 
        for c in c_nums:
            print("Cluster={}".format(c))
            print()

        
            X_tr = vect.fit_transform(train['abstract'])
            #X_tr = X_tr.todense()
            #pca = PCA(n_components=3).fit(X_tr)
            svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
            svd.fit(X_tr)
            data2D = svd.transform(X_tr)
            new_train = pd.DataFrame(columns=train.columns)
            model = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=2000)
            model_ori = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=2000)
            
            model.fit(data2D)
            model_ori.fit(X_tr)
            #i = 0
            
            new_train = pd.concat([train.reset_index(drop=True), pd.DataFrame(data2D)], axis=1)
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
            new_train['cluster'] = model.labels_
            #model_ori = KMeans(n_clusters=c)
            #model_ori.fit(X_dense)
            #model_ori.fit(X_tr)
            #print("Prediction X_test ", model_ori.predict(X_dense))
            #print(model_ori.predict(X_te))
            X_te = vect.transform(test['abstract'])
            #print("Xtr.shape",X_tr.shape)
            #print("train.shape", train.shape)
            #print("Xte.shape",X_te.shape)
            #print("test.shape", test.shape)

            
            #print("Xte.shape after transform",X_te.shape)
            test_pred = model_ori.predict(X_te)
            terms = vect.get_feature_names()
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
          
def main():
    results_dir = "/home/maleeha/research/results/"
    #results_dir = "/Users/Mal/Desktop/results/"
    
    #results_dir = "/Users/Mal/Documents/results/"

    if(os.path.exists(results_dir)==0):
        os.mkdir(results_dir)
    #path = '/Users/Mal/Desktop/sqlite/research.db'
    #path = '/Users/Mal/Documents/research.db'
    path = '/home/maleeha/research/research.db'
    process_entries(path, results_dir, 1)
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

def kmeans_cluster(df, results_dir):
    #link
    #print("Stats of run_{r}:".format(r=run))
    print("Stats:")
    r =  df['citation_count'].max() - df['citation_count'].min()
    print("\trange of citation count:", r)

    Qmin = np.min(df['citation_count'])
    Q1 = np.percentile(df['citation_count'], 25, interpolation = 'midpoint') 
    Q3 = np.percentile(df['citation_count'], 75, interpolation = 'midpoint') 
    Q2 = np.median(df['citation_count'])
    Qmax = np.max(df['citation_count'])
    
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
    #file = open(output_dir + "output_{r}.txt".format(r=run), "w")
    # file.write("Data Stats:\n")
    # file.write("\trange of citation count:" + str(df['citation_count'].max() - df['citation_count'].min()) + "\n")
    # file.write("\tmin\t" + str(Qmin) + "\n")
    # file.write("\tQ1\t" + str(Q1) + "\n")
    # file.write("\tQ2\t" + str(Q2) + "\n")
    # file.write("\tQ3\t" + str(Q3) + "\n")
    # file.write("\tmax\t" + str(Qmax) + "\n")
    # file.close()
    #plt.figure(figsize=(5,5))
    #df['citation_count'].plot(kind='density')
    #plt.show()
    vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
    #X = vect.fit_transform(df['abstract']) #.iloc[:80]
    train, test = train_test_split(df, train_size=0.75)
    #print(train.shape, test.shape)
    #print(test)
    #vocab = vect.vocabulary_
    #idf = vect.idf_
    #print(vect.get_feature_names())
    #print(X.shape)
    model = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=2000)
    i = 0
    while count <= len(train['abstract']):
        try:
            mini_df = train['abstract'][i:i+2000]
            model = mini_kmeans(vect, mini_df, model)
            i += 2000
        except IndexError:
            break
    #len(train['abstract'])
    
    #X_tr = vect.fit_transform(train['abstract'])
    #X_te = vect.transform(test['abstract'])
    #X_tr = X_tr.todense()
    #X_te = X_te.todense()
    #print("train test split - different transform ", X_tr.shape, X_te.shape)
    #X_dense = X.todense() # link
    #print(X_dense.shape)
    #X_train = X_dense[:80]
    #X_test = X_dense[80:]
    #print("X_dense split - same transform ", X_train.shape, X_test.shape)
    # ncomponents = 3
    pca = PCA(n_components=3).fit(X_tr)
    #pca = PCA(n_components=3).fit(X_train)
    data2D = pca.transform(X_tr)
    #data2D = pca.transform(X_train)

    terms = vect.get_feature_names()
    c_nums = [3, 6, 8, 10, 15, 25] # changed 50 to 40 bc of convergence warnings
    #c_nums = [3]
    print("K-Means Clustering =====>")
    print()
    #file_string = "figures_{r}.pdf".format(r=run) 
    #with PdfPages(figure_dir + "figures_run{r}.pdf".format(r=run)) as pdf: 
    with PdfPages(results_dir + "figures.pdf") as pdf: 
        for c in c_nums:
            print("Cluster={}".format(c))
            print()
            
            # file = open(results_dir + "c_terms.csv", "a")
            # file.write("Cluster={}\n".format(c))
            # file.close()
            model = KMeans(n_clusters=c)
            model.fit(data2D)
            #print("model labels ", model.labels_, model.labels_.shape)
        
            #df_new = pd.concat([df.reset_index(drop=True), pd.DataFrame(data2D)], axis=1)
            df_new = pd.concat([train.reset_index(drop=True), pd.DataFrame(data2D)], axis=1)
            df_new['cluster'] = model.labels_
                
            model_ori = KMeans(n_clusters=c)
            #model_ori.fit(X_dense)
            model_ori.fit(X_tr)
            #print("Prediction X_test ", model_ori.predict(X_dense))
            #print(model_ori.predict(X_te))
            test_pred = model_ori.predict(X_te)
            #print("Prediction scores on test data ", test_pred)
            #print(test.loc[0])
            test_dict = {}
            for i in range(len(X_te)):
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

            df_new.rename(columns={0: 'Component 1', 1 : 'Component 2', 2: 'Component 3'}, inplace=True)
            means = df_new.groupby(['cluster']).mean()
            means['citation_count'] = means['citation_count'].apply(trunc)
            #print("means", list(means.index))
            #means['cluster'] = means.index
            
            test_pred_df = pd.DataFrame(columns=list(df.columns) + [ 'predicted_cluster', 'expected_cit_count', 'pred_str'])
            #test_pred_df = pd.DataFrame(columns=['bibcode', 'title', 'predicted_cluster', 'expected_cit_count', 'pred_str'])
            print(test_pred_df.columns)
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
            f = plt.figure()
            x_axis = df_new['Component 2']
            y_axis = df_new['Component 1']
            #plt.figure(figsize=(10,8))
            ax = sns.scatterplot(x_axis, y_axis, hue = df_new['cluster'], palette=list(cluster_colors.values()))
            ax.legend(fontsize=6)
            xs = [p[1] for p in centers2D] # link
            ys = [p[0] for p in centers2D]
            #print(xs, ys)
            print(cluster_terms_ori.items())
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
            x_axis = df_new['Component 3']
            y_axis = df_new['Component 2']
            #plt.figure(figsize=(10,8))
            ax = sns.scatterplot(x_axis, y_axis, hue = df_new['cluster'], palette=list(cluster_colors.values()))
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
            x_axis = df_new['Component 3']
            y_axis = df_new['Component 1']
            #plt.figure(figsize=(10,8))
            ax = sns.scatterplot(x_axis, y_axis, hue = df_new['cluster'], palette=list(cluster_colors.values()))
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
 
