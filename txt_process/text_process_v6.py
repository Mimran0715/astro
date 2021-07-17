import warnings
warnings.filterwarnings("ignore")
#import pdfplumber
import re # add regular expression to take care of special characters and hyphens?
import sqlite3 
from matplotlib.backends.backend_pdf import PdfPages
import time
import os

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
import seaborn as sns 
#365
sns.set() 
#365
paper_count = 3769874 

#vocab = dict()

#def process_entries(path, output_dir, test_pred_dir, figure_dir):
def process_entries(path, results_dir):
    global paper_count
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    #cur.execute("SELECT count(*) FROM astro_papers_2;")
    #paper_count = cur.fetchone()[0]
    #print(type(cur.fetchone()[0]))
    #conn.close()
    run = 0 
    #while True:
    df_1 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 1980 AND year <1990 LIMIT 2500;", conn)
    #kmeans_cluster(df, run, output_dir, test_pred_dir, figure_dir)
        #run += 10
    df_2 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 1990 AND year <2000 LIMIT 2500;", conn)
    #kmeans_cluster(df, run, output_dir, test_pred_dir, figure_dir)
        #run += 10
    df_3 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 2000 AND year <2010 LIMIT 2500;", conn)
    #kmeans_cluster(df, run, output_dir, test_pred_dir, figure_dir)
        #run += 10
    df_4 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 2010 AND year <2022 LIMIT 2500;", conn)
    #df = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers;", conn)
    df = pd.concat([df_1, df_2, df_3, df_4], axis=0)
    kmeans_cluster(df, results_dir)
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

#def kmeans_cluster(df,run, output_dir, test_pred_dir, figure_dir):
def kmeans_cluster(df, results_dir):
    #link
    #print("Stats of run_{r}:".format(r=run))
    print("Stats:")
    print("\trange of citation count:", df['citation_count'].max() - df['citation_count'].min())

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

    file = open(results_dir + "output.txt", "w")
    #file = open(output_dir + "output_{r}.txt".format(r=run), "w")
    file.write("Data Stats:\n")
    file.write("\trange of citation count:" + str(df['citation_count'].max() - df['citation_count'].min()) + "\n")
    file.write("\tmin\t" + str(Qmin) + "\n")
    file.write("\tQ1\t" + str(Q1) + "\n")
    file.write("\tQ2\t" + str(Q2) + "\n")
    file.write("\tQ3\t" + str(Q3) + "\n")
    file.write("\tmax\t" + str(Qmax) + "\n")
    file.close()
    #plt.figure(figsize=(5,5))
    #df['citation_count'].plot(kind='density')
    #plt.show()
    
    vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))

    X = vect.fit_transform(df['abstract']) #.iloc[:80]
    train, test = train_test_split(df, train_size=0.75)
    #print(train.shape, test.shape)
    #print(test)
    #vocab = vect.vocabulary_
    #idf = vect.idf_
    #print(vect.get_feature_names())
    #print(X.shape)
    X_tr = vect.fit_transform(train['abstract'])
    X_te = vect.transform(test['abstract'])
    X_tr = X_tr.todense()
    X_te = X_te.todense()
    #print("train test split - different transform ", X_tr.shape, X_te.shape)
    X_dense = X.todense() # link
    #print(X_dense.shape)
    X_train = X_dense[:80]
    X_test = X_dense[80:]
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
            
            file = open(results_dir + "output.txt", "a")
            file.write("Cluster={}\n".format(c))
            file.close()
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
            cluster_terms_ori = []
            for i in range(c):
                c_terms = " ".join([terms[ind] for ind in order_centroids[i, :c]])
                cluster_terms.append(c_terms)
                c_terms_ori = " ".join([terms[ind] for ind in order_centroids_ori[i, :c]])
                cluster_terms_ori.append((i, c_terms_ori))

            file = open(results_dir + "output.txt", "a")
            #file = open(output_dir + "output_run{r}.txt".format(r=run), "a")
            #print("Cluster Terms")
            file.write("\tCluster Terms\n")
            for x in cluster_terms_ori:
                #print("\t" + str(x[0]) , x[1])
                file.write("\t" + str(x[0]) + "\t" + x[1] + "\n")
            #print()
            #print("cluster terms ori ", cluster_terms_ori)
            file.close()

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
            if c == 3:
                for i, txt in enumerate(cluster_terms_ori): # link
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
                for i, txt in enumerate(cluster_terms_ori): # link
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
                for i, txt in enumerate(cluster_terms_ori): # link
                    text = "    " + str(i) + " " + txt[1]
                    plt.annotate(text, (xs[i], ys[i]), (xs[i], ys[i]),fontsize='medium',c='k', weight='bold') # link
                plt.scatter(centers2D[:,2], centers2D[:,0], marker='*', s=125, linewidths=3, c='k')
            plt.title("{x}-means Clusters - PCA components 3 v. 1".format(x=c))
            plt.show()
            pdf.savefig(f3)
            #f.savefig("component3v1.pdf", bbox_inches='tight')
            #print("done making plots ....")
def main():
    output_dir = "/home/maleeha/research/output_txt/"
    test_pred_dir = "/home/maleeha/research/test_pred/"
    
    figure_dir = "/home/maleeha/research/plots/"
    results_dir = "/home/maleeha/research/results/"

    if(os.path.exists(output_dir)==0):
        os.mkdir(output_dir)
    if(os.path.exists(test_pred_dir)==0):
        os.mkdir(test_pred_dir)
    if(os.path.exists(figure_dir)==0):
        os.mkdir(figure_dir)
    if(os.path.exists(results_dir)==0):
        os.mkdir(results_dir)
    #path = '/Users/Mal/Desktop/sqlite/research.db'
    #path = '/Users/Mal/Desktop/sqlite/research.db'
    path = '/home/maleeha/research/research.db'
    df = process_entries(path, results_dir)
    #df = process_text(df)
    #kmeans_cluster(df, output_dir_, test_pred_dir, figure_dir)
    #t0 = time.time()
    #plot_words(df) 
    #print("time elapsed for plot words ", time.time()-t0)
    #agg_cluster(df)
   
if __name__ == '__main__':
    main()
