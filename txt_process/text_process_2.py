#import pdfplumber
import re # add regular expression to take care of special characters and hyphens?
import sqlite3 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance 

from nltk import FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from collections import defaultdict
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
import random

import seaborn as sns #365
sns.set()

from scipy.cluster.vq import vq, kmeans

vocab = defaultdict(int)
vocab_tokens = defaultdict(int)
vocab_stems = defaultdict(int)
vocab_bigrams = defaultdict(int)
vocab_trigrams = defaultdict(int)

# create a vocabulary class with a Posting class

# class Vocabulary:
#     def __init__(self):
#         self.index = {}
#         self.word_count = {}
#         self.count = 0

#     def add_word():
#         pass

# Going into the file system and extracting the text of each pdf and storing it in BLOB/TEXT?
#---------------------------------x------------------------------
# def obtain_pdf_text(file_path):
#     #for file in listdir('/Users/Mal/Desktop/research'):
#     #    if file.endswith(".pdf"):
#     #        print("------------------------------x------------------------------")
#     pdf = pdfplumber.open(file_path)
#     paper_text = ""
#     for page in pdf.pages:
#         text = page.extract_text()
#         paper_text += text
#     #db_command('''INSERT INTO astro_papers(paper_text) VALUES(?);''', (paper_text,))
#     pdf.close()
#     return paper_text

#def remove_chars(text):
#     pass
#     #return re.sub("")

#CODE STARTS HERE ---------------------------------x------------------------------

def parse_text(text):
    return re.findall("([^(.*)][a-zA-Z0-9']{2,})", text)

# process text
def read_entries(path):
    conn = sqlite3.connect(path)
    df = pd.read_sql_query("SELECT  bibcode, title, year, author, abstract, citation_count FROM astro_papers_2 LIMIT 100;", conn)
    #df = pd.read_sql_query("SELECT bibcode, title, citation_count, abstract FROM astro_papers_2 LIMIT 5;", conn)
    conn.close()
    return df

def tokenize(text):
    return [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]

def tokenize_bigram(text):
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    return list(ngrams(tokens, 2))

def tokenize_trigram(text):
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    return list(ngrams(tokens, 3))

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

    
    tokenized = []
    for abs_text in df['abs_text']:
        abstract_str = " ".join(abs_text)
        tokenized.append(abstract_str)

    df['abs_tokenized'] = tokenized

    for abs_text in abs_text_token:
        for term in abs_text:
            vocab_tokens[term] +=1 

    for abs_text in abs_text_stem:
        for term in abs_text:
            vocab_stems[term] +=1 

    for abs_text in abs_text_bigram:
        for term in abs_text:
            vocab_bigrams[term] += 1


    #abstracts = df['abs_text'].tolist()
    #print(len(df['abs_text']))
    #print(df.head())
    #print(df['abs_text'][0])

    #vocab = []
    #for i in range(df.shape[0]):
        #vocab.extend(df['abs_text'][i])
        #rint(len(vocab))
    #for term in abs_text_token:
    #    vocab_tokens.update

    return df

def trunc(number):
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

def plot_words(df):
    years = []
    for i in range(1980, 2022):
        years.append(i)

    counts = pd.DataFrame(index=vocab_tokens, columns=years)
    #print(counts)

    # counts.loc[token] = count_1980 count_1981 ...
    for i in range(1980, 2022):
        d = df[df['year'] == i]
        for token in vocab_tokens:
            arr = []
            year_count = 0
            for index, row in d.iterrows():
                year_count += word_count(token, row['abs_tokenized'])
            arr.append(year_count)
        
        counts.loc[token] = arr
        #arr.append(year_count)
        #print(arr)
        #counts.loc[token] = arr


    print(counts)

    #for index, row in df.iterrows(): #https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
        #new_dict = {}

        # vocab_count = defaultdict(int)
        # for year in years:
        #     if row['year'] == year:
        #         for token in vocab_tokens:
        #             for term in row['abs_tokenized']:
        #                 if token == term:
        #                     vocab_count[token] +=1
        #     new_dict.update({token: vocab_count[token]})
        #counts = counts.append(new_dict, ignore_index =True) #https://www.geeksforgeeks.org/how-to-create-an-empty-dataframe-and-append-rows-columns-to-it-in-pandas/

    counts = counts.fillna(0) #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
    #counts = counts[counts[counts.columns] > 1]
    #print(counts)
    #pass

def label_point(x, y, val, ax): #https://stackoverflow.com/questions/46027653/adding-labels-in-x-y-scatter-plot-with-seaborn
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))

def kmeans_cluster(df):
    #link 
    #df = df.sample(frac=1).reset_index(drop=True)
    tfidfvect = TfidfVectorizer(stop_words='english')
    X = tfidfvect.fit_transform(df['abs_tokenized'])
    X = X.todense() # link
    #pca = PCA(n_components=4).fit(X) # link
    #pca = PCA(n_components=4).fit(X)
    #pca = PCA().fit(X) #n_components == min(n_samples, n_features) - 1 from sklearn website

    #data2D = pca.transform(X)
    
    #link
    #print(X.shape)
    #print(data2D.shape)
    #print(pca.explained_variance_ratio_)

    # plt.figure(figsize= (10, 8))
    # plt.plot(range(1,6), pca.explained_variance_ratio_.cumsum(), marker='o',linestyle='--')
    # plt.title("Explained Variance by Components")
    # plt.xlabel("Number of Components")
    # plt.ylabel("Cumulative Explained Variance")
    # plt.show()

    # ncomponents = 3
    pca = PCA(n_components=3).fit(X)
    data2D = pca.transform(X)

    # wcss = []
    # for i in range(1, 6):
    #     kmeans_pca = KMeans(n_clusters=i, init = 'k-means++', random_state=42)
    #     kmeans_pca.fit(pca_scores)
    #     wcss.append(kmeans_pca.inertia_)

    # plt.figure(figsize= (10, 8))
    # plt.plot(range(1,6), wcss, marker='o',linestyle='--')
    # plt.title("Number of Clusters")
    # plt.xlabel("WCSS")
    # plt.ylabel("K-means with PCA Clustering")
    # plt.show()
    
    #clusters = 4
    #plt.hist(tfidfvect.idf_)
    #plt.show()
    
    terms = tfidfvect.get_feature_names()
    c_nums = [3, 6, 8, 10, 15, 25] # changed 50 to 40 bc of convergence warnings
    print("K-Means Clustering =====>")
    print()
    #true_k = 6

    for c in c_nums:
        print("Cluster={}".format(c))
        model = KMeans(n_clusters=c, init='k-means++', max_iter=200, n_init=10)
        model.fit(data2D)

        df_new = pd.concat([df.reset_index(drop=True), pd.DataFrame(data2D)], axis=1)
        # print(df_new.columns)
        # df_new['cluster'] = model.labels_
               
        model_ori = KMeans(n_clusters=c, init='k-means++', max_iter=200, n_init=10)
        model_ori.fit(X)

        order_centroids = model.cluster_centers_.argsort()[:, ::-1]  #only give PCA values
        order_centroids = model_ori.cluster_centers_.argsort()[:, ::-1] 
        cluster_terms = []
        for i in range(c):
            #print("Cluster %d: " % i, end='')
            #for ind in order_centroids[i, :c]:
            #    print('%s ' % terms[ind], end='')
            c_terms = " ".join([terms[ind] for ind in order_centroids[i, :c]])
            cluster_terms.append(c_terms)
            #print()

        df_new['terms'] = cluster_terms

        df_new.rename(columns={0: 'Component 1', 1 : 'Component 2', 2: 'Component 3'}, inplace=True)
        xs = df_new['Component 2']
        ys = df_new['Component 1']
        df_new['cluster'] = model.labels_

        cluster_colors = dict()
        for l in model.labels_:
            cluster_colors[l] = (random.random(), random.random(), random.random())


        x_axis = df_new['Component 2']
        y_axis = df_new['Component 1']
        plt.figure(figsize=(10,8))
        ax = sns.scatterplot(x_axis, y_axis, hue = df_new['cluster'], palette=list(cluster_colors.values()))
        plt.title("Cluster by PCA components 2 v. 1")
        label_point(x_axis, y_axis, df_new['terms'], plt.gca())  
        plt.show()

        x_axis = df_new['Component 3']
        y_axis = df_new['Component 2']
        plt.figure(figsize=(10,8))
        ax = sns.scatterplot(x_axis, y_axis, hue = df_new['cluster'], palette=list(cluster_colors.values()))
        plt.title("Cluster by PCA components 3 v. 2")
        label_point(x_axis, y_axis, df_new['terms'], plt.gca())  
        plt.show()

        x_axis = df_new['Component 3']
        y_axis = df_new['Component 1']
        plt.figure(figsize=(10,8))
        ax = sns.scatterplot(x_axis, y_axis, hue = df_new['cluster'], palette=list(cluster_colors.values()))
        plt.title("Cluster by PCA components 3 v. 1")
        label_point(x_axis, y_axis, df_new['terms'], plt.gca())  
        plt.show()

        # model_ori = KMeans(n_clusters=c, init='k-means++', max_iter=200, n_init=10)
        # model_ori.fit(X)

        #labels=model.labels_
        #df_2=pd.DataFrame(list(zip(df['title'],labels, df['citation_count'])),columns=['title','cluster', 'citation_count'])
        
        #print(df_2.sort_values(by=['cluster']))
        #print(model.cluster_centers_)
        #print(model.inertia_)

        # means = df_2.groupby(['cluster']).mean()
        # means['citation_count'] = means['citation_count'].apply(trunc)
        #     #print(means.columns)

        # print(means)
        # print()

        # #df_print = pd.DataFrame(dict(x=data2D[:,0], y=data2D[:,1], label=df_2['cluster']))
        # df_print = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=df_2['cluster']))
        # groups = df_print.groupby('label')
        # print(groups)
        # cluster_colors = dict()
        # for n in range(c):
        #     cluster_colors[n] = (random.random(), random.random(), random.random())
        # #cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
        # for name, group in groups:
        #     plt.scatter(group.x, group.y,color=cluster_colors[name])
    
        # #centers2D_o = pca.transform(model.cluster_centers_)
        # centers2D = model.cluster_centers_
        # #print("C ori", centers2D_o.shape)
        # print("C", centers2D.shape)
        # if(c ==3):
        #     x_axis = [p[0] for p in centers2D] # link
        #     y_axis = [p[1] for p in centers2D]
        #     for i, txt in enumerate(cluster_terms): # link
        #         text = str(i) + " " + txt
        #         plt.annotate(text, (x_axis[i], y_axis[i]), (x_axis[i], y_axis[i]),fontsize='medium',c='k') # link
        # plt.scatter(centers2D[:,0], centers2D[:,1], marker='*', s=150, linewidths=3, c='r')
        # plt.show()  # link
        # print()
    
    #link
def agg_cluster(df):
    #link
    tfidfvect = TfidfVectorizer(stop_words='english')
    X = tfidfvect.fit_transform(df['abs_tokenized'])
    X = X.todense() # link

    pca = PCA(n_components=3).fit(X) # link
    data2D = pca.transform(X)
    
    terms = tfidfvect.get_feature_names()
    
    C = 1 - cosine_similarity(X.T)
    pca_cosine = PCA(n_components=3).fit(C) # link
    dataC = pca_cosine.transform(C)
    num_clusters = 6
    c_nums = [3 ,6, 8, 10, 15, 25] 
    links = ["ward", "complete", "average", "single"]
    for c in c_nums:
        for link in links:
            model_ori = AgglomerativeClustering(n_clusters=c, linkage=link)
            model_ori.fit(X)

            model = AgglomerativeClustering(n_clusters=c, linkage=link)
            model.fit(data2D)

            labels=model_ori.labels_
            df_2=pd.DataFrame(list(zip(df['title'], labels, df['citation_count']))\
                              ,columns=['title','cluster', 'citation_count'])
            
            print("Euclidean -> Cluster={} , linkage = {}".format(c,link))
            means = df_2.groupby(['cluster']).mean()
            means['citation_count'] = means['citation_count'].apply(trunc)
            #print(means.columns)
            print(means)
            print()
            #             model_cos = AgglomerativeClustering(n_clusters=num_clusters, linkage=link).fit(C)
            #             labels_cos = model_cos.labels_
            #             df_2=pd.DataFrame(list(zip(df['title'], labels_cos, df['citation_count']))\
            #                               ,columns=['title','cluster', 'citation_count'])
                        
            #             print("COS SIM MATRIX - > Cluster={} , linkage = {}".format(c,link))
            #             print(df_2.groupby(['cluster']).mean())
            #             print()
            Z = hierarchy.linkage(model_ori.children_, link)
            plt.figure(figsize=(20,10))
            dn = hierarchy.dendrogram(Z)
            plt.show()
    
        for link in links[1:]:
            model = AgglomerativeClustering(n_clusters=c, affinity='cosine', linkage=link)
            model.fit(X)
            labels=model.labels_
            df_2=pd.DataFrame(list(zip(df['title'], labels, df['citation_count']))\
                              ,columns=['title','cluster', 'citation_count'])
            
            print("Cosine - > Cluster={} , linkage = {}".format(c,link))
            means = df_2.groupby(['cluster']).mean()
            means['citation_count'] = means['citation_count'].apply(trunc)
            #print(means.columns)
            print(means)
            print()

def kmeans_cluster_old(df):
    #below taken from link 
    df = df.sample(frac=1).reset_index(drop=True)
    
    train, test = train_test_split(df, test_size=0.2)
    tfidfvect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    #print(tfidfvect.vocabulary_)
    
    X = tfidfvect.fit_transform(df['abs_tokenized'])

    print(tfidfvect.vocabulary_)
    '''
    xs, ys = X[:, 0], X[:, 1]
    
    first_vector = X[0]
    dataframe = pd.DataFrame(first_vector.T.todense(), index = tfidfvect.get_feature_names(), columns = ["tfidf"])
    dataframe.sort_values(by = ["tfidf"],ascending=False)
    print(dataframe["tfidf"])
    #link
    
    #link
    terms = tfidfvect.get_feature_names()
    c_nums = [3, 6, 8, 10, 15, 25] # changed 50 to 40 bc of convergence warnings
    print("K-Means Clustering =====>")
    true_k = 6
    for c in c_nums:
        print("Number of clusters ", c)
        model = KMeans(n_clusters=c, init='k-means++', max_iter=200, n_init=10)
        model.fit(X)
        labels=model.labels_
        df_2=pd.DataFrame(list(zip(df['title'],labels, df['citation_count'])),columns=['title','cluster', 'citation_count'])
        #print(df_2.sort_values(by=['cluster']))
        #print(model.cluster_centers_)
        df_3 = pd.DataFrame(dict(x=xs, y=ys, label=labels, title=df['title'])) 
        order_centroids = model.cluster_centers_.argsort()[:, ::-1] # copy
        group = df_3.groupby('cluster')
        for i in range(c):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :c]:
                print(' %s' % terms[ind], end='')
            print()
        
        #print(df_2[df_2['cluster']==0])
        
        #plt.scatter(df_2[df_2['cluster']==0], df_2[df_2['cluster']==1])
        #plt.show()
    #link
    
    #link
    # svd = TruncatedSVD(n_components=30, random_state=42)
    # X = svd.fit_transform(X)
    # print(f"Total variance explained: {np.sum(svd.explained_variance_ratio_):.2f}")
    '''

def agg_cluster_old(df):

    #link
    tfidfvect = TfidfVectorizer(stop_words='english')
    X = tfidfvect.fit_transform(df['abs_tokenized'])

    first_vector = X[0]
    dataframe = pd.DataFrame(first_vector.T.todense(), index = tfidfvect.get_feature_names(), columns = ["tfidf"])
    dataframe.sort_values(by = ["tfidf"],ascending=False)
    #link
  
    #link
    #terms = tfidfvect.get_feature_names();
    #link
    num_clusters = 6
    c_nums = [3, 6, 8, 10, 15, 25, 50]
    links = ["ward", "complete", "average", "single"]
    #print("Agglomerative Clustering directly on X")
    for c in c_nums:
        for link in links:
            model = AgglomerativeClustering(n_clusters=c, linkage=link)
            model.fit(X.toarray())
            labels=model.labels_
            df_2=pd.DataFrame(list(zip(df['title'],labels, df['citation_count'])),columns=['title','cluster', 'citation_count'])
            #print("Cluster={x} , linkage = {y}", c, link)
            #print(df_2.groupby(['cluster']).mean())
    #print("Agglomerative Clustering on cosine similarity")
    C = 1 - cosine_similarity(X.T)
    model = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward').fit(C)
    #link
    #model = AgglomerativeClustering(n_clusters=6, affinity="cosine", linkage='average')
    #model.fit(X.toarray())
    
    #link
    # can cluster the similarities as well
    #print("dist", dist)

def main():
    path = '/Users/Mal/Desktop/sqlite/research.db'
    df = read_entries(path)
    df = process_text(df)
    #plot_words(df)
    kmeans_cluster(df)
    #agg_cluster(df)
   
if __name__ == '__main__':
    main()
    