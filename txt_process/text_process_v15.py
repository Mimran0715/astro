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
#import pdfplumber

from nltk import FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

#from sklearn.cluster import KMeans
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.cluster import AgglomerativeClustering
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
#from sklearn import feature_extraction # br
#from sklearn.decomposition import TruncatedSVD
#from sklearn.svm import SVC
#from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer # lsi
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split

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

from sklearn.cluster import DBSCAN
import random
import seaborn as sns 
from pprint import pprint # lda link
#import pyLDAvis
#import pyLDAvis.gensim

#from scipy.cluster.hierarchy import dendrogram
#from scipy.cluster import hierarchy
#import pyLDAvis
#import pyLDAvis.sklearn
#pyLDAvis.enable_notebook()
sns.set() #365 link
paper_count = 3769874 

#vocab = dict()

# def get_model_topics(model, vectorizer, topics, n_top_words=15): # towardsdatascience article 
#     word_dict = {}
#     feature_names = vectorizer.get_feature_names()
#     for topic_idx, topic in enumerate(model.components_):
#         top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
#         top_features = [feature_names[i] for i in top_features_ind]
#         #word_dict[topics[topic_idx]] = top_features
#         word_dict[topic_idx] = top_features

#     return pd.DataFrame(word_dict)

# def get_inference(model, vectorizer, topics, text, threshold): # towards data science article
#     v_text = vectorizer.transform([text])
#     score = model.transform(v_text)

#     labels = set()
#     for i in range(len(score[0])):
#         if score[0][i] > threshold:
#             labels.add(topics[i])

#     if not labels:
#         return 'None', -1, set()

#     return topics[np.argmax(score)], score, labels
#def process_entries(path, output_dir, test_pred_dir, figure_dir):

def process_entries(path, results_dir, folder_dir, size, cluster):
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
        kmeans_clustering(df, results_dir, folder_dir)
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
        kmeans_clustering(df, results_dir, folder_dir)
    elif size == 2:
        df_1 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 1980 AND year <1990 LIMIT 100;", conn)
        df_2 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 1990 AND year <2000 LIMIT 100;", conn)
        df_3 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 2000 AND year <2010 LIMIT 100;", conn)
        df_4 = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 WHERE year >= 2010 AND year <2022 LIMIT 100;", conn)
        df = pd.concat([df_1, df_2, df_3, df_4], axis=0)
        print("DF INFO --------->")
        print(df.info(verbose=False, memory_usage="deep"))
        print()
        kmeans_clustering(df, results_dir, folder_dir)
 
    conn.close()
    #return df

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
    #abs_text = abs_text.apply(parse_text)
    #print(abs_text) 

    abs_text = abs_text.apply(parse_text)
    abs_text = abs_text.apply(' '.join)

    #abs_text = abs_text.apply(stemmer.stem())

    abs_text_stem = abs_text.apply(tokenize_and_stem)
    abs_text_token = abs_text.apply(tokenize)
    abs_text_bigram = abs_text.apply(tokenize_bigram)
    #abs_text_trigram = abs_text.apply(tokenize_trigram)

    #print(df.columns)
    #print(abs_text_bigram)
    #df['abs_text'] = abs_text_stem #? 
    df['abs_text'] = abs_text_token
    #df['abs_text'] = abs_text_bigram
    #sia = SentimentIntensityAnalyzer()

    
    # tokenized = []
    # for abs_text in df['abs_text']:
    #     abstract_str = " ".join(abs_text)
    #     tokenized.append(abstract_str)

    # df['abs_tokenized'] = tokenized

    # for abs_text in abs_text_token:
    #     for term in abs_text:
    #         vocab_tokens[term] +=1 

    # for abs_text in abs_text_stem:
    #     for term in abs_text:
    #         vocab_stems[term] +=1 

    # for abs_text in abs_text_bigram:
    #     for term in abs_text:
    #         vocab_bigrams[term] += 1
    return df

def trunc(number): #so
    text = f"{number:.3f}"
    return float(text)

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

def see_clusters(model, data, pdf):
    f = plt.figure()
    visualizer = SilhouetteVisualizer(model,  colors='yellowbrick')

    visualizer.fit(data)       # Fit the data to the visualizer
    #visualizer.set_title(title)
    visualizer.show() 
    pdf.savefig(f)

def get_sent(df):

    sia = SentimentIntensityAnalyzer()
    neg_scores = []
    neu_scores = []
    pos_scores = []
    comp_scores = []
    abs_text = df['abstract']
    for text in abs_text:
        neg_scores.append(sia.polarity_scores(text)['neg'])
        neu_scores.append(sia.polarity_scores(text)['neu'])
        pos_scores.append(sia.polarity_scores(text)['pos'])
        comp_scores.append(sia.polarity_scores(text)['compound'])
    df['abs_sent_neg'] = neg_scores
    df['abs_sent_neu'] = neu_scores
    df['abs_sent_pos'] = pos_scores
    df['abs_sent_comp'] = comp_scores
    return df

def get_lda(df):
    df_2 = process_text(df)
    #df_2 = get_sent(df_2)
    print(df_2.head())
    train_2, test_2 = train_test_split(df_2, train_size=0.75)
    temp = df
    #df = df_2
    #getting data

    texts = list(train_2['abs_text'])
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

    #plot_sent(df_2)
    #pyLDAvis.enable_notebook()
    #vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

    lda = LatentDirichletAllocation(n_components=7, random_state=1)
    #lda.fit(data)
    lda.fit(X_tr_count)
    for i,topic in enumerate(lda.components_):
        print(f'Top 10 words for topic #{i}:')
        print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
        print('\n')
    pprint(lda_model.print_topics())

    #data_LDA_tr = lda.transform(X_tr_count)
    #data_LDA_te = lda.transform(X_te_count)

# model_no == 0 --> kmeans, model_no == 1 --> dbscan
def kmeans_clustering(df, results_dir, kmeans_dir):    
    # preprocessing

    # print("Stats:")
    # r =  df['citation_count'].max() - df['citation_count'].min()
    # print("\trange of citation count:", r)

    Qmin = np.min(df['citation_count'])
    Q1 = np.percentile(df['citation_count'], 25, interpolation = 'midpoint') 
    Q3 = np.percentile(df['citation_count'], 75, interpolation = 'midpoint') 
    Q2 = np.median(df['citation_count'])
    Qmax = np.max(df['citation_count'])

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

    vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))
    count_vect = CountVectorizer(stop_words='english',strip_accents='unicode',token_pattern=r"[a-zA-Z0-9']{2,}", ngram_range=(1,2))

    df = get_sent(df)
    train, test = train_test_split(df, train_size=0.75)

    X_tr_tfidf = vect.fit_transform(train['abstract'])
    tfidf_terms = vect.get_feature_names()
    X_tr_count = count_vect.fit_transform(train['abstract'])
    count_terms = count_vect.get_feature_names()
    X_te_tfidf = vect.transform(test['abstract'])
    X_te_count = count_vect.transform(test['abstract'])

    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    data2D_tr = svd.fit_transform(X_tr_tfidf)
    data2D_te = svd.transform(X_te_tfidf)


    # print("Data Shapes:")
    # print("X_tr_tfidf shape: ", X_tr_tfidf.shape)
    # print("X_tr_count shape: ", X_tr_count.shape)
    # print("X_te_tfidf shape: ", X_te_tfidf.shape)
    # print("X_te_count shape: ", X_te_count.shape)
    # print()
    c_nums = [3, 6, 8, 10, 15, 25]
    
    #clustering
    cluster_plots_pdf = PdfPages(results_dir + "cluster_plots.pdf")
    test_pred_pdf = PdfPages(results_dir + "test_pred.pdf")
    silhouette_pdf = PdfPages(results_dir + "silhouette.pdf")
    #df = temp
    #with PdfPages(results_dir + "figures.pdf") as pdf: 
    with PdfPages(results_dir + "cluster_plots.pdf") as cluster_plots_pdf,\
        PdfPages(results_dir + "test_pred.pdf") as test_pred_pdf,\
        PdfPages(results_dir + "silhouette.pdf") as silhouette_pdf:
        for c in c_nums:
            print("Cluster={}".format(c))
            print()
        
            model_p = kmeans_dir + "model_" + str(c) + ".pkl"
            model_ori_p = kmeans_dir + "model_ori_" + str(c) + ".pkl"

            model = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=2000)
            model_ori = MiniBatchKMeans(n_clusters=c,random_state=0,batch_size=2000)

            model_ori.fit(X_tr_tfidf)
            #see_clusters(model_ori, X_tr_tfidf, "model_ori", silhouette_pdf)
            pickle.dump(model_ori, open(model_ori_p,'wb')) # machine learning mastery

            #data2D = topic_model(0, X_tr_tfidf)   
            model.fit(data2D_tr)
            see_clusters(model, data2D_tr, silhouette_pdf)
            pickle.dump(model, open(model_p, 'wb'))

            new_train = pd.DataFrame(columns=train.columns)
            new_train = pd.concat([train.reset_index(drop=True), pd.DataFrame(data2D_tr)], axis=1)
            new_train['cluster'] = model.labels_
            
            test_pred = model_ori.predict(X_te_tfidf)
            test_pred_data2D = model.predict(data2D_te)
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
   
# def dbs_cluster(df, results_dir):
#     folder_str = results_dir + "/dbs/"
#     model = DBSCAN(eps=3, min_samples=2)

           
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

def plot_sent(df):
    plt.figure()
    df['abs_sent_neg'].plot(title='neg scores', kind='kde')
    plt.show()
    df['abs_sent_neu'].plot(title='neu scores', kind='kde')
    print
    plt.show()
    df['abs_sent_pos'].plot(title='pos scores', kind='kde')
    plt.show()
    df['abs_sent_comp'].plot(title='comp scores', kind='kde')
    plt.show()

    print("Stats: neg")
    r =  df['abs_sent_neg'].max() - df['abs_sent_neg'].min()
    print("\trange of citation count:", r)

    Qmin = np.min(df['abs_sent_neg'])
    Q1 = np.percentile(df['abs_sent_neg'], 25, interpolation = 'midpoint') 
    Q3 = np.percentile(df['abs_sent_neg'], 75, interpolation = 'midpoint') 
    Q2 = np.median(df['abs_sent_neg'])
    Qmax = np.max(df['abs_sent_neg'])

    print("\tmin: ", Qmin)
    print("\tQ1: ", Q1)
    print("\tQ2: ", Q2)
    print("\tQ3: ", Q3)
    print("\tmax: ", Qmax)
    print()

    print("Stats: neu")
    r =  df['abs_sent_neu'].max() - df['abs_sent_neu'].min()
    print("\trange of citation count:", r)

    Qmin = np.min(df['abs_sent_neu'])
    Q1 = np.percentile(df['abs_sent_neu'], 25, interpolation = 'midpoint') 
    Q3 = np.percentile(df['abs_sent_neu'], 75, interpolation = 'midpoint') 
    Q2 = np.median(df['abs_sent_neu'])
    Qmax = np.max(df['abs_sent_neu'])

    print("\tmin: ", Qmin)
    print("\tQ1: ", Q1)
    print("\tQ2: ", Q2)
    print("\tQ3: ", Q3)
    print("\tmax: ", Qmax)
    print()

    print("Stats: pos")
    r =  df['abs_sent_pos'].max() - df['abs_sent_pos'].min()
    print("\trange of citation count:", r)

    Qmin = np.min(df['abs_sent_pos'])
    Q1 = np.percentile(df['abs_sent_pos'], 25, interpolation = 'midpoint') 
    Q3 = np.percentile(df['abs_sent_pos'], 75, interpolation = 'midpoint') 
    Q2 = np.median(df['abs_sent_pos'])
    Qmax = np.max(df['abs_sent_pos'])

    print("\tmin: ", Qmin)
    print("\tQ1: ", Q1)
    print("\tQ2: ", Q2)
    print("\tQ3: ", Q3)
    print("\tmax: ", Qmax)
    print()

    print("Stats: comp")
    r =  df['abs_sent_comp'].max() - df['abs_sent_comp'].min()
    print("\trange of citation count:", r)

    Qmin = np.min(df['abs_sent_neu'])
    Q1 = np.percentile(df['abs_sent_comp'], 25, interpolation = 'midpoint') 
    Q3 = np.percentile(df['abs_sent_comp'], 75, interpolation = 'midpoint') 
    Q2 = np.median(df['abs_sent_comp'])
    Qmax = np.max(df['abs_sent_comp'])

    print("\tmin: ", Qmin)
    print("\tQ1: ", Q1)
    print("\tQ2: ", Q2)
    print("\tQ3: ", Q3)
    print("\tmax: ", Qmax)
    print()

    #for sent in df['abs_sent']:

def plot_results(path):
    #path = '/Users/Mal/Documents/results'
    with PdfPages("results.pdf") as pdf: 
        for filename in os.listdir(path):
            if "test_pred" in filename:
                name = os.path.join(path, filename)
                f = plt.figure()
                d = pd.read_csv(name, error_bad_lines=False) # skipping bad entries 
                #d = pd.read_csv(name)
                print(d.head())
                print(d.columns)
                #ax = sns.histplot(data=pd.read_csv(name), x='year', hue='pred_str')
                ax = sns.histplot(data=d, x='year', hue='pred_str')
                ax.set_title("File: " + name + " test prediction results")
                plt.show()
                pdf.savefig(f)
    
def main():
    #results_dir = "/home/maleeha/research/results/"
    run_type = sys.argv[1]
    results_dir = "/Users/Mal/Desktop/results/"
    kmeans_dir = "/Users/Mal/Desktop/results/kmeans/"
    #results_dir = "/Users/Mal/Documents/results/"

    if(os.path.exists(results_dir)==0):
        os.mkdir(results_dir)

    if(os.path.exists(kmeans_dir)==0):
        os.mkdir(kmeans_dir)
    # if(os.path.exists(kmeans_dir)==0):
    #     os.mkdir(kmeans_dir)
    folder_dir = kmeans_dir # kmeans == 0
    
    #path = '/Users/Mal/Desktop/sqlite/research.db'
    path = '/Users/Mal/Documents/research.db'
    #path = '/home/maleeha/research/research.db'
    #process_entries(path, results_dir, 0)
    process_entries(path, results_dir, folder_dir, int(run_type), 0)
    #path = '/Users/Mal/Desktop/resu'
    #plot_results(results_dir)
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