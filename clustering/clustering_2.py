import warnings
warnings.filterwarnings("ignore")
import re 
import sqlite3 
from matplotlib.backends.backend_pdf import PdfPages
import time
import os
import csv
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.ion() # take away popups - putting plot images in pdf

from nltk import FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import nltk
#from nltk.sentiment import SentimentIntensityAnalyzer

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer # lsi
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
#from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA

from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from gensim.models import LsiModel
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import Sparse2Corpus
from gensim.corpora import Dictionary

from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

import random
import seaborn as sns 
from pprint import pprint # lda link

from utils import get_data, db_command, get_tbl_count

#for getting stats
citation_range = 0
Qmin = 0
Qmax = 0
Q1 = 0
Q2 = 0
Q3 = 0

def trunc(number): # taken from stack overflow to print avg citation count
    text = f"{number:.3f}"
    return float(text)

def see_clusters(model, data, pdf):
    f = plt.figure()
    visualizer = SilhouetteVisualizer(model,  colors='yellowbrick')
    visualizer.fit(data) # Fit the data to the visualizer
    #visualizer.set_title(title)
    visualizer.show() 
    pdf.savefig(f)

def get_lda_gensim(df):
    texts = list(df['abs_text'])
    tagged = []
    for text in texts:
        tag = nltk.pos_tag(text)
        tagged.append(tag)
    print(tagged[0])
    id2word = Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus=corpus,
                        id2word=id2word,
                        num_topics=10, 
                        random_state=100,
                        update_every=1,
                        chunksize=100,
                        passes=10,
                        alpha='auto',
                        per_word_topics=True)
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    pprint(lda_model.print_topics())
    #pyLDAvis.enable_notebook()
    #vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

def kmeans_training(df, cluster_num, model, transformer, vect):
    
    #train, test = train_test_split(df, train_size=0.80) #80/20 train test split
    #vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
    #X_tr_tfidf = vect.fit_transform(df['abstract'])
    #tfidf_terms = vect.get_feature_names()
    #X_te_tfidf = vect.transform(test['abstract'])

    # with PdfPages(results_dir + "cluster_plots.pdf") as cluster_plots_pdf,\
    #     PdfPages(results_dir + "test_pred.pdf") as test_pred_pdf,\
    #     PdfPages(results_dir + "silhouette.pdf") as silhouette_pdf, \
    #     PdfPages(results_dir + "wordcloud.pdf") as wordcloud_pdf:
        #for c in c_nums:

    #incrementally run on training data then get testing data at once and then do?? 
    model = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=df.shape[0])
    
    transformer.partial_fit(train)
    PCA_data = transformer.transform(train)
    model.partial_fit(PCA_data)

    #see_clusters(model, PCA_data, silhouette_pdf)
    #pickle.dump(model, open(model_path, 'wb'))

    #model_ori_p = kmeans_dir + "model_ori_" + str(c) + ".pkl"
    #model_ori = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=2000)
    #model_ori.fit(X_tr_tfidf)
    #see_clusters(model_ori, X_tr_tfidf, "model_ori", silhouette_pdf)
    #pickle.dump(model_ori, open(model_ori_p,'wb')) # machine learning mastery  
    return model, transformer      

def kmeans_testing(df,vect,model, transformer):
    # new_train = pd.DataFrame(columns=train.columns)
    # new_train = pd.concat([train.reset_index(drop=True), pd.DataFrame(data2D_tr)], axis=1)
    # new_train['cluster'] = model.labels_
    
    X_te_tfidf = ''

    test_pred = model_ori.predict(X_te_tfidf)
    #test_pred_data2D = model.predict(data2D_te)
    #print("Test Pred _ X_te_tfidf")
    #print(test_pred)
    #print("ORI")
    silhouette_avg = silhouette_score(X_te_tfidf, test_pred) #scikitlearn silhouette
    #print("For n_clusters =", c, 
    #        "The average silhouette_score is :", silhouette_avg)
    #print()
    #print("DATA2D")
    silhouette_avg = silhouette_score(data2D_te, test_pred_data2D) #scikitlearn silhouette
    #print("For n_clusters =", c, 
    #       "The average silhouette_score is :", silhouette_avg)

    #print("Prediction scores on test data ", test_pred)
    #print(test.loc[0])
    test_dict = {}
    for i in range(X_te_tfidf.shape[0]):
        #print("updating test_dict ...at value {x} out of {y}".format(x=i, y=len(X_te)))
        curr = test.iloc[i]
        #print(curr)
        #break
        # print(curr, test_pred[i])
        # print(type(curr.values))
        # print(type(tuple(curr.values)))
        #x = tuple(curr.values)
        #print(x)
        # print(type(x))
        #print(type(x), type(test_pred[i]))
        test_dict.update({tuple(curr.values):test_pred[i]})
        #test_dict.update({tuple(curr.values):test_pred[i]})
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
        c_terms = " ".join([tfidf_terms[ind] for ind in order_centroids[i, :c]])
        cluster_terms.append(c_terms)
        c_terms_ori = " ".join([tfidf_terms[ind] for ind in order_centroids_ori[i, :c]])
        #cluster_terms_ori.append((i, c_terms_ori))
        cluster_terms_ori[i] = c_terms_ori

    c_file = open(results_dir + "c_terms" + str(c) + ".csv", "w")
    #stats = {'range':r, "min":Qmin, "q1":Q1, "q2":Q2, "q3":Q3, "max":Qmax}
    
    writer = csv.writer(c_file) #kite
    for key, value in cluster_terms_ori.items():
        writer.writerow([key, value])
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(value)
        #plt.figure()
        wf = plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        wordcloud_pdf.savefig(wf)
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
    print("means.columns", means.columns)
    #print(means.head())

    means['citation_count'] = means['citation_count'].apply(trunc)
    means['abs_sent_neg'] = means['abs_sent_neg'].apply(trunc)
    means['abs_sent_neu'] = means['abs_sent_neu'].apply(trunc)
    means['abs_sent_pos'] = means['abs_sent_pos'].apply(trunc)
    means['abs_sent_comp'] = means['abs_sent_comp'].apply(trunc)

    print(means.head())
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
    
        try:
            # print("testing...")
            # print("bibcode: ", x[0][0])
            # print("title: ",x[0][1])
            # print("year: ", x[0][2])
            # print("author: ",x[0][3])
            # print("abstract: ", x[0][4])
            # print("citation_count: ", x[0][5])
            # print("predicted_cluster: ", x[1])
            # print("expected_cit_count: ", exp_cit_count)
            # print("pred_str: ", pred_str)
            # print()

            test_pred_df = test_pred_df.append({'bibcode':x[0][0], 'title':x[0][1], \
                'year':x[0][2], 'author':x[0][3], 'abstract':x[0][4], 'citation_count':x[0][5],\
                    'predicted_cluster':x[1],'expected_cit_count':exp_cit_count, 'pred_str':pred_str},\
                        ignore_index=True)
        except IndexError:
            #print(len(x))
            arr.append(len(x))
            test_pred_df = test_pred_df.append({'bibcode':x[0][0], 'title':x[0][1], \
                'predicted_cluster':x[1],\
            'expected_cit_count':exp_cit_count, 'pred_str':pred_str}, ignore_index=True)
            #pass
        #print(x[0], "predicted cluster: ", x[1], "avg citation count of cluster {x}: ".format(x=x[1]), m['citation_count'])
        #print(m)
        p_count+=1

    # print("test pred pdf head")
    # print(test_pred_df.head())
    # print()
    #print(set(arr))
    
    #print("Cluster Prediction of Test Set Plus Expected Citation Count")
    #print()
    #print(test_pred_df)
    #print()
    #test_file = results_dir + "test_pred" + str(c) + ".csv"
    #test_file = results_dir + "test_pred" + str(c) + "run_{r}".format(r=run) + ".csv"
    #test_pred_df.to_csv(test_file)

    #with PdfPages("results.pdf") as pdf: 
            #name = os.path.join(path, filename)
    #print(test_pred_df.columns)
    f = plt.figure()
    # d = pd.read_csv(name, error_bad_lines=False) # skipping bad entries 
    #         #d = pd.read_csv(name)
    #         print(d.head())
    #         print(d.columns)
            #ax = sns.histplot(data=pd.read_csv(name), x='year', hue='pred_str')
    ax = sns.histplot(data=test_pred_df, x='year', hue='pred_str')
    ax.set_title("File: cluster_" + str(c) + " test prediction results")
    plt.show()
    test_pred_pdf.savefig(f)

    cluster_colors = dict()
    for l in model.labels_:
        cluster_colors[l] = (random.random(), random.random(), random.random())
    #print("creating plot 1 ....")
    #print(centers2D)

    plot_clusters(cluster_plots_pdf, c, new_train, 0, cluster_colors, centers2D, cluster_terms_ori)
    plot_clusters(cluster_plots_pdf, c, new_train, 1, cluster_colors, centers2D, cluster_terms_ori)
    plot_clusters(cluster_plots_pdf, c, new_train, 2, cluster_colors, centers2D, cluster_terms_ori)
    

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

def process_entries(df, results_dir, folder_dir, run_type):
    if run_type == 0:
        kmeans_clustering(df, results_dir, folder_dir)
    elif run_type == 1:
        get_lda(df)

def get_curr_stats(df):
    global citation_range, Q1, Q2, Q3, Qmin, Qmax

    print("Stats:")
    r =  df['citation_count'].max() - df['citation_count'].min()
    print("\trange of citation count:", r)

    qmin = np.min(df['citation_count'])
    q1 = np.percentile(df['citation_count'], 25, interpolation = 'midpoint') 
    q3 = np.percentile(df['citation_count'], 75, interpolation = 'midpoint') 
    q2 = np.median(df['citation_count'])
    qmax = np.max(df['citation_count'])

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

def main():
    #df = ""
    results_dir = "/Users/Mal/Documents/results/"
    folder_dir = "/Users/Mal/Documents/results/kmeans/"
    db_path = '/Users/Mal/Documents/research.db'

    run_loc = sys.argv[1]
    tbl_name = sys.argv[2]
    batch_size = sys.argv[3]

    if run_loc == 1:
        results_dir = "/home/maleeha/research/results/"
        folder_dir = "/home/maleeha/research/results/kmeans/"
        db_path = '/home/maleeha/research/research.db'

    if(os.path.exists(results_dir)==0):
        os.mkdir(results_dir)

    if(os.path.exists(folder_dir)==0):
        os.mkdir(folder_dir)

    curr_size = 0
    entry_count = get_tbl_count(db_path, tbl_name)
    data_split = int(0.80 * entry_count)

    count_vect = CountVectorizer(stop_words='english',strip_accents='unicode',token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
    lda_model = LatentDirichletAllocation(n_components=7, random_state=1)

    while curr_size < data_split:
        df =  get_data(db_path, tbl_name, batch_size)
        X_tr = count_vect.fit_transform(df['abstract'])
        lda_model.partial_fit(X_tr)
        curr_size += batch_size
    
    #count_terms = count_vect.get_feature_names()
    
    for i,topic in enumerate(lda_model.components_):
        print(f'Top 10 words for topic #{i}:')
        print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
        print('\n')

    while curr_size < data_split:
        df =  get_data(db_path, tbl_name, batch_size)
        X_tr = count_vect.fit_transform(df['abstract'])
        lda_model.partial_fit(X_tr)
        curr_size += batch_size

    curr_size = 0
    while curr_size < data_split:
        df =  get_data(db_path, tbl_name, batch_size)
        X_tr = count_vect.fit_transform(df['abstract'])
        lda_model.partial_fit(X_tr)
        curr_size += batch_size


    data_LDA_tr = lda_model.transform(X_tr_count)
    #data_LDA_te = lda_model.transform(X_te_count)

    kmeans_model = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=df.shape[0])

    while curr_size < data_split:
        kmeans_model.partial_fit(data_LDA_tr)
        curr_size += batch_size

    # # training kmeans
    # cluster_val = 8 # test and change to optimum

    # curr_size = 0
    # entry_count = get_tbl_count(db_path, tbl_name)
    # data_split = int(0.80 * entry_count)

    # transformer = IncrementalPCA(n_components=7, batch_size=batch_size) 
    # while curr_size != data_split:
    #     df = get_data(db_path, tbl_name, batch_size)
    #     transformer.partial_fit(df)
    
    # curr_size = 0
    # while curr_size != data_split:
    #     df = get_data(db_path, tbl_name, batch_size)
    #     transformer.transform(df)
    

    #     #model, transformer = kmeans_training(df, cluster_val, model, transformer)
    #     #get_lda(df)
    #     curr_size += size

    # model_path = kmeans_dir + "model_" + str(cluster_val) + ".pkl"

if __name__ == "__main__":
    main()

# def plot_results(path):
#     #path = '/Users/Mal/Documents/results'
#     with PdfPages("results.pdf") as pdf: 
#         for filename in os.listdir(path):
#             if "test_pred" in filename:
#                 name = os.path.join(path, filename)
#                 f = plt.figure()
#                 d = pd.read_csv(name, error_bad_lines=False) # skipping bad entries 
#                 #d = pd.read_csv(name)
#                 print(d.head())
#                 print(d.columns)
#                 #ax = sns.histplot(data=pd.read_csv(name), x='year', hue='pred_str')
#                 ax = sns.histplot(data=d, x='year', hue='pred_str')
#                 ax.set_title("File: " + name + " test prediction results")
#                 plt.show()
#                 pdf.savefig(f)


# def get_sent(df):
#     sia = SentimentIntensityAnalyzer()
#     neg_scores = []
#     neu_scores = []
#     pos_scores = []
#     comp_scores = []
#     abs_text = df['abstract']
#     for text in abs_text:
#         neg_scores.append(sia.polarity_scores(text)['neg'])
#         neu_scores.append(sia.polarity_scores(text)['neu'])
#         pos_scores.append(sia.polarity_scores(text)['pos'])
#         comp_scores.append(sia.polarity_scores(text)['compound'])
#     df['abs_sent_neg'] = neg_scores
#     df['abs_sent_neu'] = neu_scores
#     df['abs_sent_pos'] = pos_scores
#     df['abs_sent_comp'] = comp_scores
#     return df

# def stats(df):
#     stats_dict = {}

#     Qmin = np.min(df['citation_count'])
#     Q1 = np.percentile(df['citation_count'], 25, interpolation = 'midpoint') 
#     Q3 = np.percentile(df['citation_count'], 75, interpolation = 'midpoint') 
#     Q2 = np.median(df['citation_count'])
#     Qmax = np.max(df['citation_count'])

#     stats_dict.update({"Qmin": Qmin})
#     stats_dict.update({"Q1":Q1})
#     stats_dict.update({"Q3":Q3})
#     stats_dict.update({"Q3":Q3})
#     stats_dict.update({"Qmax":Qmax})
    
#     return stats_dict

# def get_data_set(vect_no, data, test):
#     if vect_no == 0:
#         vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
#         d = vect.fit_transform(data)
#         terms = vect.get_feature_names()
#         return (d, terms)

#     elif vect_no == 1:
#         count_vect = CountVectorizer(stop_words='english',strip_accents='unicode',token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
#         dc = count_vect.fit_transform(data)
#         terms = count_vect.get_feature_names()
#         return (dc, terms)

#     #terms = vect.get_feature_names()

# no == 0 --> lda, no == 1 --> lsi, 
# def topic_model(tm_no, data):
#     if tm_no == 0:
#         svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
#         svd.fit(data)
#         data2D = svd.transform(data)
#         return data2D

#     elif tm_no == 1:
#         lda = LatentDirichletAllocation(n_components=2, random_state=1)
#         lda.fit(data)
#         data_LDA = lda.transform(data)
#         return data_LDA
#     #X_tr = X_tr.todense()
#             #pca = PCA(n_components=3).fit(X_tr)
#             #d = pd.DataFrame(X_tr_c.toarray(),index=train['abstract'],columns=count_vect.get_feature_names()) # assn pdf
#             #print(d.head())
#             #X_tr_c = X_tr_c.todense()
#             #print(X_tr_c)
#             #print(len(list(count_vect.get_feature_names())))
#             #print(len(count_vect.vocabulary))
#             #corpus_vect = Sparse2Corpus(X_tr_c, documents_columns=False)
#             #print(count_vect.vocabulary)
#             #lsa_model = LsiModel(corpus_vect, id2word=count_vect.vocabulary)
#             #print(lsa_model.print_topics())

#def vect_data(df, vect_type):
#     train, test = train_test_split(df, train_size=0.80) #80/20 train test split
#     if vect_type == 0: # tfidf
#         vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
#         X_tr_tfidf = vect.fit_transform(train['abstract'])
#         tfidf_terms = vect.get_feature_names()
#         X_te_tfidf = vect.transform(test['abstract'])
#         return (train, test, vect, X_tr_tfidf, tfidf_terms, X_te_tfidf)
#     elif vect_type == 1: #count
#         count_vect = CountVectorizer(stop_words='english',strip_accents='unicode',token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
#         X_tr_count = count_vect.fit_transform(train['abstract'])
#         count_terms = count_vect.get_feature_names()
#         X_te_count = count_vect.transform(test['abstract'])
#         return (train, test, count_vect, X_tr_count, count_terms, X_te_count)
