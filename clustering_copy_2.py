import math
import sqlite3
import warnings
import math
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
import random
import seaborn as sns 
import re
from tabulate import tabulate
warnings.filterwarnings("ignore")

from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

def parse_text(text):
    return re.findall("[a-zA-Z0-9']{2,}", text)

def tokenizer_1(text:str):
    # print("In tokenizer_1, init text:", text)
    text = parse_text(text)
    # print("After parse_text:", text)
    text = " ".join(text)
    # print("After join text:", text)
    
    stemmer = SnowballStemmer('english')
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    stems = [s for s in stems if 'sub' not in s and 'sup' not in s]
    return stems # taken from: http://brandonrose.org/clustering_mobile
    
def see_clusters(model, data, pdf):
    f = plt.figure()
    visualizer = SilhouetteVisualizer(model,  colors='yellowbrick')
    visualizer.fit(data) # Fit the data to the visualizer
    #visualizer.set_title(title)
    visualizer.show() 
    pdf.savefig(f)

def plot_clusters(pdf, c, new_train, ax_val, cluster_colors, centers2D):
    f = plt.figure()
    if ax_val == 0:
        x_axis = new_train['Component 2']
        y_axis = new_train['Component 1']
        title = "TruncSVD Components 2 v. 1"
    elif ax_val == 1:
        x_axis = new_train['Component 3']
        y_axis = new_train['Component 2']
        title = "TruncSVD Components 3 v. 2"
    elif ax_val == 2:
        x_axis = new_train['Component 3']
        y_axis = new_train['Component 1']
        title = "TruncSVD Components 3 v. 1"
    
    #plt.figure(figsize=(10,8))
    ax = sns.scatterplot(x_axis, y_axis, hue = new_train['cluster'], palette=list(cluster_colors.values()))
    ax.legend(fontsize=6)
    xs = [p[1] for p in centers2D] # link
    ys = [p[0] for p in centers2D]
    #print(xs, ys)
    #print(cluster_terms_ori.items())
    # if c == 3:
    #     for i, txt in enumerate(cluster_terms_ori.items()): # link
    #         #print(txt[1], type(txt[1]))
    #         text = "    " + str(i) + " " + txt[1]
    #         plt.annotate(text, (xs[i], ys[i]), (xs[i], ys[i]),fontsize='medium',c='k', weight='bold') #link
    #     plt.scatter(centers2D[:,1], centers2D[:,0], marker='*', s=125, linewidths=3, c='k')
    plt.title(("{x}-means Clusters -" + title).format(x=c))
    plt.show()
    pdf.savefig(f)

# def check_year(year):
#     if year >= 1980 and year < 1985:
#         return "1980-1984"
#     elif year >=1985 and year < 1990:
#         return "1985-1989"
#     elif year >=1990 and year < 1995:
#         return "1990-1994"
#     elif year >=1995 and year < 2000:
#         return "1995-1999"
#     elif year >=2000 and year < 2005:
#         return "2000-2004"
#     elif year >=2005 and year < 2010:
#         return "2005-2009"
#     elif year >=2010 and year < 2015:
#         return "2010-2014"
#     elif year >=2015:
#         return "2015-2021"

def check_year_2(year):
    if year >= 1980 and year < 1985:
        return 0
    elif year >=1985 and year < 1990:
        return 1
    elif year >=1990 and year < 1995:
        return 2
    elif year >=1995 and year < 2000:
        return 3
    elif year >=2000 and year < 2005:
        return 4
    elif year >=2005 and year < 2010:
        return 5
    elif year >=2010 and year < 2015:
        return 6
    elif year >=2015:
        return 7

def fix_arr(arr):
    min_arr = min(arr)
    max_arr = max(arr)
    for x in arr:
        x = (x-min_arr) / (max_arr-min_arr)
    return arr

def violin_plot(df, pdf):
    #print("Now doing.. violin plot")
    f = plt.figure(figsize=(10,8))
    sns.set_theme(style="whitegrid")
    sns.violinplot(x=df['year_d'],y=df['SVD_comp_0'],inner='quartile')
    plt.title("comp 0 v Year- VIOLIN")
    plt.ylabel("SVD comp 0")
    plt.xlabel("Year")
    plt.show()
    pdf.savefig(f)

def kde_plot(df, pdf):
    pass

def main():
    #db_path = input("DB path: ")
    #sample_size = input("Sample size: ")
    #c = int(input("Number of clusters: "))
    #n_components = int(input("Number of components for SVD: "))

    run_loc = sys.argv[1]  # local == 0, clotho == 1

    if run_loc == 0:
        db_path = '/Users/Mal/Documents/research.db'
    else:
        db_path = '/home/maleeha/research/code/research.db'

    c = 8
    n_components = 3

    tbl_name = input("Table name: ")
    runs = int(input("Number of runs: "))

    conn = sqlite3.connect(db_path)
    #df = pd.read_sql("SELECT * FROM " + str(tbl_name) + " ORDER BY RANDOM() LIMIT " + str(sample_size) + ";", conn)
    df = pd.read_sql("SELECT * FROM " + str(tbl_name) + ";", conn)
    conn.close()
    print("got data")

    for run in range(runs):
        r = random.randint(0, run)
        df_mini = df.sample(n=10000,random_state=r) # fix random state
        print("run...{x}".format(x=run))

        train, test = train_test_split(df_mini, train_size=0.80)
        #vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}",\
        #    ngram_range=(1,2))
        vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', tokenizer=tokenizer_1, ngram_range=(1,2))
        X_tr = vect.fit_transform(train['abstract'])
        X_te = vect.transform(test['abstract'])

        #print("X_tr.shape:", X_tr.shape)
        #print(X_tr[0])
        #break
        #pca = PCA(n_components=2)   # PCA does not support sparse input
        # n_vals = [2, 3, 5, 8, 15, 25, 50, 100, 250, 500, 1000]
        # for n in n_vals:
         #     var_ratio = svd.explained_variance_ratio_
        #     total_var = 0.0
        #     for v in var_ratio:
        #         total_var += v
        #     print("N val:", n, "total_var", total_var)
        # #print(select_n_components(var_ratio, 0.95))
        # break
        # print(df_mini.shape)
         #print(svd_data[0])
        #break
        #print("Shape:", svd_data.shape)
        #print("First Entry:", svd_data[0])
        #sorted_array = svd_data[np.argsort(-svd_data[:, 4])]  # kite - https://www.kite.com/python/answers/how-to-sort-the-rows-of-a-numpy-array-by-a-column-in-python
        #print("First entry after sorting:", sorted_array[0])
        #print(sorted_array[:10])
        #break


        svd = TruncatedSVD(n_components=n_components)
        data2D_tr = svd.fit_transform(X_tr)
        data2D_te = svd.transform(X_te)

        svd_data = np.zeros((data2D_tr.shape[0], data2D_tr.shape[1]+2))
        for i in range(data2D_tr.shape[0]):
            #print(df_mini.iloc[i]['year'])
            #print(df_mini.iloc[i]['citation_count'])
             #print(arr)
            arr = np.append(data2D_tr[i], [df_mini.iloc[i]['year'], df_mini.iloc[i]['citation_count']])
            svd_data[i] = arr
       
        df = pd.DataFrame(data=svd_data, columns=["SVD_comp_0", "SVD_comp_1","SVD_comp_2", "year", "citation_count"])
        df['year_d'] = df.apply(lambda x:check_year_2(x['year']), axis=1)
        df['citation_count'] = df.apply(lambda x: math.log(x['citation_count'], 10), axis=1)
        #print(df.groupby("year").to_frame())
        #print(df.head())
        #return

        with PdfPages("svd_plots_" + str(run) + ".pdf") as pdf:
            # kde plot of year vs component
            # print("kde plot of year vs component")
            # f = plt.figure()
            # sns.kdeplot(x=df['year_d'], y=df['SVD_comp_0'], cmap="Blues", shade=True, bw_adjust=.5) #pyplot gallery
            # plt.title("SVD Component 0 vs Year")
            # plt.xlabel("Year")
            # plt.ylabel("SVD comp 0")
            # plt.show()
            # pdf.savefig(f)

            # f = plt.figure()
            # sns.kdeplot(x=df['year_d'], y=df['SVD_comp_1'], cmap="Blues", shade=True, bw_adjust=.5) #pyplot gallery
            # plt.title("SVD Component 1 vs Year")
            # plt.xlabel("Year")
            # plt.ylabel("SVD comp 1")
            # plt.show()
            # pdf.savefig(f)

            # f = plt.figure()
            # sns.kdeplot(x=df['year_d'], y=df['SVD_comp_2'], cmap="Blues", shade=True, bw_adjust=.5) #pyplot gallery
            # plt.title("SVD Component 2 vs Year")
            # plt.xlabel("Year")
            # plt.ylabel("SVD comp 2")
            # plt.show()
            # pdf.savefig(f)

            # density plot of component vs cit count
            print("density plot of component vs cit count")
            f = plt.figure()
            ax = plt.gca()
            #ax.set_yscale('log')
            #ax.set_yticks([1, 10, 100, 1000,10000])
            #ax.set_yticks([1, 10, 100, 1000, 5000, 10000])
            ax.scatter(x=df['SVD_comp_0'], y=df['citation_count'])
            #sns.kdeplot(x=df['SVD_comp_0'], y=df['citation_count'], cmap="Blues", shade=True, bw_adjust=.5, ax=ax) #pyplot gallery
            plt.title("SVD Component 0 vs Citation_Count")
            plt.xlabel("SVD Comp 0")
            plt.ylabel("Citation Count")
            plt.show()
            pdf.savefig(f)

            f = plt.figure()
            ax = plt.gca()
            #ax.set_yscale('log')
            #ax.set_yticks([1, 10, 100, 1000,10000])
            sns.kdeplot(x=df['SVD_comp_0'], y=df['citation_count'], cmap="Blues", shade=True, bw_adjust=.5, ax=ax) #pyplot gallery
            plt.title("SVD Component 0 vs Citation_Count")
            plt.xlabel("SVD Comp 0")
            plt.ylabel("Citation Count")
            plt.show()
            pdf.savefig(f)

            f = plt.figure()
            ax = plt.gca()
            #ax.set_yscale('log')
            #ax.set_yticks([1, 10, 100, 1000,10000])
            sns.kdeplot(x=df['SVD_comp_0'], y=df['citation_count'], cmap="Blues", shade=True, bw_adjust=.5, ax=ax) #pyplot gallery
            plt.title("SVD Component 0 vs Citation_Count")
            plt.xlabel("SVD Comp 0")
            plt.ylabel("Citation Count")
            plt.show()
            pdf.savefig(f)
            
            # violin plot of year vs component
            print("violin plot of year vs component")
            f = plt.figure(figsize=(10,8))
            sns.set_theme(style="whitegrid")
            sns.violinplot(x=df['year_d'],y=df['SVD_comp_0'],inner='quartile')
            plt.title("comp 0 v Year- VIOLIN")
            plt.ylabel("SVD comp 0")
            plt.xlabel("Year")
            plt.show()
            pdf.savefig(f)

            f = plt.figure(figsize=(10,8))
            sns.set_theme(style="whitegrid")
            sns.violinplot(x=df['year_d'],y=df['SVD_comp_1'],inner='quartile')
            plt.title("comp 1 v Year- VIOLIN")
            plt.ylabel("SVD comp 1")
            plt.xlabel("Year")
            plt.show()
            pdf.savefig(f)

            f = plt.figure(figsize=(10,8))
            sns.set_theme(style="whitegrid")
            sns.violinplot(x=df['year_d'],y=df['SVD_comp_2'],inner='quartile')
            plt.title("comp 2- VIOLIN")
            plt.ylabel("SVD comp 2")
            plt.xlabel("Year")
            plt.show()
            pdf.savefig(f)
            
            #log citation vs svd
            print("log citation vs svd")
            f = plt.figure()
            ax = plt.gca()
            #x_axis = svd_data[:, 0]
            #y_axis = svd_data[:, 4]
            #ax.scatter(x_axis, y_axis)
            ax.scatter(df['SVD_comp_0'], df['citation_count'])
            ax.set_yscale('log')
            ax.set_yticks([1, 10, 100, 1000,10000])
            plt.title("SVD Component 0 vs Citation_Count - FOR DEMO set yscale log")
            plt.xlabel("SVD comp 0")
            plt.ylabel("Cit Count")
            plt.show()
            pdf.savefig(f)

            f = plt.figure()
            ax = plt.gca()
            # x_axis = svd_data[:, 1]
            # y_axis = svd_data[:, 4]
            # ax.scatter(x_axis, y_axis)
            ax.scatter(df['SVD_comp_1'], df['citation_count'])
            ax.set_yscale('log')
            ax.set_yticks([1, 10, 100, 1000,10000])
            plt.title("SVD Component 1 vs Citation_Count - FOR DEMO set yscale log")
            plt.xlabel("SVD comp 1")
            plt.ylabel("Cit Count")
            plt.show()
            pdf.savefig(f)

            f = plt.figure()
            ax = plt.gca()
            # x_axis = svd_data[:, 2]
            # y_axis = svd_data[:, 4]
            #ax.scatter(x_axis, y_axis)
            ax.scatter(df['SVD_comp_2'], df['citation_count'])
            ax.set_yscale('log')
            ax.set_yticks([1, 10, 100, 1000,10000])
            plt.title("SVD Component 2 vs Citation_Count - FOR DEMO set yscale log")
            plt.xlabel("SVD comp 2")
            plt.ylabel("Cit Count")
            plt.show()
            pdf.savefig(f)

            #violin of citation vs component
            print("violin of citation vs component")
            #f = plt.figure(figsize=(10,8))
            f = plt.figure()
            ax = plt.gca()
            ax.set_yscale('log')
            ax.set_yticks([1, 10, 100, 1000,10000])
            sns.set_theme(style="whitegrid")
            sns.violinplot(x=df['SVD_comp_0'],y=df['citation_count'],inner='quartile', ax=ax)
            plt.title("comp 0 v Year- VIOLIN")
            plt.xlabel("SVD comp 0")
            plt.ylabel("cit count")
            plt.show()
            pdf.savefig(f)

            #f = plt.figure(figsize=(10,8))
            f = plt.figure()
            ax = plt.gca()
            ax.set_yscale('log')
            ax.set_yticks([1, 10, 100, 1000,10000])
            sns.set_theme(style="whitegrid")
            sns.violinplot(x=df['SVD_comp_1'],y=df['citation_count'],inner='quartile', ax=ax)
            plt.title("comp 1 v Year- VIOLIN")
            plt.xlabel("SVD comp 1")
            plt.ylabel("cit count")
            plt.show()
            pdf.savefig(f)

            #f = plt.figure(figsize=(10,8))
            f = plt.figure()
            ax = plt.gca()
            ax.set_yscale('log')
            ax.set_yticks([1, 10, 100, 1000,10000])
            sns.set_theme(style="whitegrid")
            sns.violinplot(x=df['SVD_comp_2'],y=df['citation_count'],inner='quartile', ax=ax)
            plt.title("comp 1 v Year- VIOLIN")
            plt.xlabel("SVD comp 2")
            plt.ylabel("cit count")
            plt.show()
            pdf.savefig(f)

        # other things
        return
        model = KMeans(n_clusters=c)
        model.fit(data2D_tr)

        with PdfPages("silhouette_" + str(run) + ".pdf") as pdf:
            see_clusters(model, data2D_tr, pdf)

        new_train = pd.DataFrame(columns=train.columns)
        new_train = pd.concat([train.reset_index(drop=True), pd.DataFrame(data2D_tr)], axis=1)
        new_train['cluster'] = model.labels_

        test_pred = model.predict(data2D_te)
        silhouette_avg = silhouette_score(X_te, test_pred)

        print("Silhouette avg score(s): ", silhouette_avg)
        print()

        centers2D = model.cluster_centers_
        order_centroids = centers2D.argsort()[:, ::-1]  

        terms = vect.get_feature_names()
        
        sorted_zip = sorted(zip(svd.components_[0], terms), key=lambda x: abs(x[0]), reverse=True)
        features_0 =  ["%+0.3f*%s" % (coef, feat) for coef, feat in sorted_zip]
        features_0_str =   " ".join(features_0)
        #features_0 = features_0.argsort()[::-1]
        #print("SVD Comp 0: ", features_0[:10])
        print("SVD COMP 0")
        print(tabulate(sorted_zip[:25]))
        #results_file.write("SVD Comp 0 features: ", features_0[:10], )

        sorted_zip = sorted(zip(svd.components_[1], terms), key=lambda x: abs(x[0]), reverse=True)
        features_1 =   ["%+0.3f*%s" % (coef, feat) for coef, feat in sorted_zip]
        features_1_str = " ".join(features_1)
        #features_1 = features_1.argsort()[::-1]
        #print("SVD Comp 1: ", features_1[:10])
        print("SVD COMP 1")
        print(tabulate(sorted_zip[:25]))

        sorted_zip = sorted(zip(svd.components_[2], terms), key=lambda x: abs(x[0]), reverse=True)
        features_2 =  ["%+0.3f*%s" % (coef, feat) for coef, feat in sorted_zip]
        features_2_str = " ".join(features_2)
        #features_2 = features_2.argsort()[::-1]
        #print("SVD Comp 2: ", features_2[:10])
        print("SVD COMP 2")
        print(tabulate(sorted_zip[:25]))
        
        print()
        print()

        # cluster_terms = []
        # for i in range(c):
        #     c_terms = " ".join([terms[ind] for ind in order_centroids[i, :c]])
        #     cluster_terms.append(c_terms)
        #     print("C_terms ", c_terms)
        #     #print("Cluster {x} terms: ".format(x=i), c_terms)
        # print(cluster_terms)

        # comp_dict = {}
        # for i in range(n_components):
        #     comp_dict[i] = "Component " + str(i)

        #new_train.rename(columns=comp_dict, inplace=True)

        cluster_colors = dict()
        for label in model.labels_:
            cluster_colors[label] = (random.random(), random.random(), random.random())

        new_train.rename(columns={0: 'Component 1', 1 : 'Component 2', 2: 'Component 3'}, inplace=True)
        
        with PdfPages("cluster_plot_" + str(run) + ".pdf") as pdf:
            #print("am about to call plot_clusters + run .. {x}".format(x=run))
            plot_clusters(pdf, c, new_train, 0, cluster_colors, centers2D)
            plot_clusters(pdf, c, new_train, 1, cluster_colors, centers2D)
            plot_clusters(pdf, c, new_train, 2, cluster_colors, centers2D)

if __name__ == "__main__":
    main()

 