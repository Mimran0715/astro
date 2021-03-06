import math
import time
import os
import csv
import pickle
import nltk

import sqlite3
import warnings
import math
from matplotlib.backends.backend_pdf import PdfPages
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
from scipy.stats import kde

from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords

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

def violin_plot(x, y, pdf, title, x_label, y_label):
    #print("Now doing.. violin plot")
    f = plt.figure(figsize=(10,8))
    sns.set_theme(style="whitegrid")
    sns.violinplot(x=x,y=y,inner='quartile')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()
    pdf.savefig(f)

def kde_plot(x, y, pdf, title, x_label, y_label, log_true=False):
    #print("kde plot of year vs component")
    f = plt.figure()
    #ax = plt.gca()
    if log_true: 
        g = sns.kdeplot(x=x, y=y, cmap="Blues", shade=True, bw_adjust=.5) #pyplot gallery
        g.set_yscale('log')
        g.set_yticks([1, 10, 100, 1000, 10000])
        
    else:
        sns.kdeplot(x=x, y=y, cmap="Blues", shade=True, bw_adjust=.5) #pyplot gallery
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    pdf.savefig(f)

def scatter_plot(x, y, pdf, title, x_label, y_label):
    f = plt.figure()
    ax = plt.gca()
    # x_axis = svd_data[:, 1]
    # y_axis = svd_data[:, 4]
    # ax.scatter(x_axis, y_axis)
    ax.scatter(x, y)
    ax.set_yscale('log')
    ax.set_yticks([1, 10, 100, 1000,10000])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    pdf.savefig(f)

# def find_closest(value, val_list):

def main():
    run_loc = sys.argv[1]  # local == 0, clotho == 1

    if run_loc == 0:
        db_path = '/Users/Mal/Documents/research.db'
    else:
        db_path = '/home/maleeha/research/code/research.db'

    n_components = 3

    #tbl_name = input("Table name: ")
    #runs = int(input("Number of runs: "))
    tbl_name = "astro_papers_t3"
    runs = 1

    conn = sqlite3.connect(db_path)
    #df = pd.read_sql("SELECT * FROM " + str(tbl_name) + " ORDER BY RANDOM() LIMIT " + str(sample_size) + ";", conn)
    df = pd.read_sql("SELECT * FROM " + str(tbl_name) + ";", conn)
    conn.close()
    print("got data")

    for run in range(runs):
        r = random.randint(0, run)
        df_mini = df.sample(n=10000,random_state=r) # fix random state
        print("run...{x}".format(x=run))
        # bibcode, title author, abstract, citation_count, year 
        train, test = train_test_split(df_mini, train_size=0.80)
        #vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}",\
        #    ngram_range=(1,2))
        vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', tokenizer=tokenizer_1, ngram_range=(1,2))
        X_tr = vect.fit_transform(train['abstract'])
        X_te = vect.transform(test['abstract'])

        svd = TruncatedSVD(n_components=n_components)
        data2D_tr = svd.fit_transform(X_tr)
        #data2D_te = svd.transform(X_te)

        svd_data = np.zeros((data2D_tr.shape[0], data2D_tr.shape[1]+2))
        for i in range(data2D_tr.shape[0]):
            arr = np.append(data2D_tr[i], [df_mini.iloc[i]['year'], df_mini.iloc[i]['citation_count']])
            svd_data[i] = arr
       
        df = pd.DataFrame(data=svd_data, columns=["SVD_comp_0", "SVD_comp_1","SVD_comp_2", "year", "citation_count"])
        df['year_d'] = df.apply(lambda x:check_year_2(x['year']), axis=1)
        terms = vect.get_feature_names()
        #print('Df data')
        #print(df.head())

        #print("svd comp 0", len(svd.components_[0]))

        #print("svd comp 1", len(svd.components_[1]))

        #print("svd comp 2", len(svd.components_[2]))

        #print("df", df.shape)
        #print("2d data tr", data2D_tr.shape)

        #print("terms ", len(terms))
    
        with PdfPages("svd_plots_" + str(run) + ".pdf") as pdf:
           
            kde_plot(df['year_d'], df['SVD_comp_0'],pdf, "SVD Component 0 vs Year", "Year", "SVD comp 0")
            kde_plot(df['year_d'], df['SVD_comp_1'],pdf, "SVD Component 1 vs Year", "Year", "SVD comp 1" )
            kde_plot(df['year_d'], df['SVD_comp_2'],pdf, "SVD Component 2 vs Year", "Year", "SVD comp 2" )
            
            # kde_plot(df['SVD_comp_0'],df['citation_count'],pdf, "SVD Component 0 vs Citation Count", "SVD Comp 0", "Cit Count", log_true=True)
            # kde_plot(df['SVD_comp_1'],df['citation_count'],pdf, "SVD Component 1 vs Citation Count", "SVD Comp 1", "Cit Count", log_true=True)
            # kde_plot(df['SVD_comp_2'],df['citation_count'], pdf,"SVD Component 2 vs Citation Count", "SVD Comp 2", "Cit Count", log_true=True)
            
            scatter_plot(df['year_d'], df['SVD_comp_0'],pdf, "SVD Component 0 vs Year", "Year", "SVD comp 0")
            scatter_plot(df['year_d'], df['SVD_comp_1'],pdf, "SVD Component 1 vs Year", "Year", "SVD comp 1")
            scatter_plot(df['year_d'], df['SVD_comp_2'],pdf, "SVD Component 2 vs Year", "Year", "SVD comp 2")

            violin_plot(df['year_d'], df['SVD_comp_0'], pdf, "SVD COMP 0 vs Year", "Year", "SVD Comp 0")
            violin_plot(df['year_d'], df['SVD_comp_1'], pdf, "SVD COMP 1 vs Year", "Year", "SVD Comp 1")
            violin_plot(df['year_d'], df['SVD_comp_2'], pdf, "SVD COMP 2 vs Year", "Year", "SVD Comp 2")

            violin_plot(df['SVD_comp_0'], df['citation_count'], pdf, "SVD COMP 0 vs Cit Count", "SVD Comp 0", "cit count")
            violin_plot(df['SVD_comp_1'], df['citation_count'], pdf, "SVD COMP 1 vs Cit Count", "SVD Comp 1", "cit count")
            violin_plot(df['SVD_comp_2'], df['citation_count'], pdf, "SVD COMP 2 vs Cit Count", "SVD Comp 2 ", "cit count")

        print("Comp 0 Stats:")
        print("min", df['SVD_comp_0'].min())
        print("max", df['SVD_comp_0'].max())
        print("median", df['SVD_comp_0'].median())
        print("mean", df['SVD_comp_0'].mean())

        print("Comp 1 Stats:")
        print("min", df['SVD_comp_1'].min())
        print("max",df['SVD_comp_1'].max())
        print("median", df['SVD_comp_1'].median())
        print("mean", df['SVD_comp_1'].mean())

        print("Comp 2 Stats:")
        print("min",df['SVD_comp_2'].min())
        print("max", df['SVD_comp_2'].max())
        print("median", df['SVD_comp_2'].median())
        print("mean", df['SVD_comp_2'].mean())

        sorted_zip_0 = sorted(zip(svd.components_[0], terms), key=lambda x: abs(x[0]), reverse=True)
        #features_0 =  ["%+0.3f*%s" % (coef, feat) for coef, feat in sorted_zip]
        print("SVD COMP 0")
        print(tabulate(sorted_zip_0))
        # vals = [x[0] for x in sorted_zip_0]
        # find_close(0.01, vals)

        sorted_zip_1 = sorted(zip(svd.components_[1], terms), key=lambda x: abs(x[0]), reverse=True)
        #features_1 =  ["%+0.3f*%s" % (coef, feat) for coef, feat in sorted_zip]
        print("SVD COMP 1")
        print(tabulate(sorted_zip_1))

        sorted_zip_2= sorted(zip(svd.components_[2], terms), key=lambda x: abs(x[0]), reverse=True)
        #features_0 =  ["%+0.3f*%s" % (coef, feat) for coef, feat in sorted_zip]
        print("SVD COMP 2")
        print(tabulate(sorted_zip_2))



if __name__ == "__main__":
    main()

 