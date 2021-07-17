import pdfplumber
import pandas as pd
import sqlite3 
from nltk import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
import nltk
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Going into the file system and extracting the text of each pdf and storing it in BLOB/TEXT?
def obtain_pdf_text(file_path):
    #for file in listdir('/Users/Mal/Desktop/research'):
    #    if file.endswith(".pdf"):
    #        print("------------------------------x------------------------------")
    pdf = pdfplumber.open(file_path)
    paper_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        paper_text += text
    #db_command('''INSERT INTO astro_papers(paper_text) VALUES(?);''', (paper_text,))
    pdf.close()
    return paper_text

def parse_text(text):
    return re.findall("([^(.*)][a-zA-Z0-9']{2,})", text)

def remove_chars(text):
    pass
    #return re.sub("")

# process text
def read_entries():
    conn = sqlite3.connect('/Users/Mal/Desktop/sqlite/research.db')
    df = pd.read_sql_query("SELECT bibcode, title, citation_count, abstract FROM astro_papers LIMIT 100;", conn)
    conn.close()
    return df

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    stemmer = SnowballStemmer('english')
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def process_text(df):
    stopwords = nltk.corpus.stopwords.words('english')
    abs_text = df['abstract']
    abs_text = abs_text.str.lower() # add regular expression to take care of special characters and hyphens?]
    #abs_text = abs_text.apply(parse_text)
    #print(abs_text) 
    abs_text = abs_text.apply(parse_text)
    abs_text = abs_text.apply(' '.join)
    #abs_text = abs_text.apply(
    abs_text = abs_text.apply(tokenize_and_stem)
    #abs_text = abs_text.apply(stemmer.stem())
    #print(abs_text)
    df['abs_text'] = abs_text
    
    tokenized = []
    for abs_text in df['abs_text']:
        abstract_str = " ".join(abs_text)
        tokenized.append(abstract_str)

    df['abs_tokenized'] = tokenized

    #abstracts = df['abs_text'].tolist()
    #print(len(df['abs_text']))
    #print(df.head())
    #print(df['abs_text'][0])
    vocab = []
    for i in range(df.shape[0]):
        vocab.extend(df['abs_text'][i])
        #print(len(vocab))
    return df

def kmeans_cluster(df):
    #below taken from link 
    df = df.sample(frac=1).reset_index(drop=True)
    
    train, test = train_test_split(df, test_size=0.2)
    tfidfvect = TfidfVectorizer(stop_words='english')
    X = tfidfvect.fit_transform(df['abs_tokenized'])
    
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

def agg_cluster(df):
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
    df = read_entries()
    df = process_text(df)
    kmeans_cluster(df)
    agg_cluster(df)
   
if __name__ == '__main__':
    main()
    
'''
def main_2():
    df = read_entries()
    df = process_text(df)
    print(df['abs_tokenized'])

    #c
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['abs_tokenized'])
    terms = tfidf_vectorizer.get_feature_names()
    print(terms)

    km = KMeans(n_clusters=5)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    print(clusters)
'''