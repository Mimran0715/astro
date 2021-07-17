import sqlite3
import warnings
warnings.filterwarnings("ignore")
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

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

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
    #f.savefig("component2v1.pdf", bbox_inches='tight')
    pdf.savefig(f)

def check_year(year):
    if year >= 1980 and year < 1990:
        return 0
    elif year >=1990 and year < 2000:
        return 1
    elif year >=2000 and year < 2010:
        return 2
    else:
        return 3

def main():

    run_loc = sys.argv[1]  # local == 0, clotho == 1

    if run_loc == 0:
        db_path = '/Users/Mal/Documents/research.db'
    else:
        db_path = '/home/maleeha/research/code/research.db'

    #db_path = input("DB path: ")
    #sample_size = input("Sample size: ")
    #c = int(input("Number of clusters: "))
    #n_components = int(input("Number of components for SVD: "))
    c = 8
    n_components = 3

    tbl_name = input("Table name: ")
    runs = int(input("Number of runs: "))

    #print("path given: ", db_path)
    #print("path type: ", type(db_path))

    conn = sqlite3.connect(db_path)

    #df = pd.read_sql("SELECT * FROM " + str(tbl_name) + " ORDER BY RANDOM() LIMIT " + str(sample_size) + ";", conn)
    df = pd.read_sql("SELECT * FROM " + str(tbl_name) + ";", conn)
    conn.close()

    #inspecting df  
    #print(df.shape)
    # year_type = df.apply(lambda x: check_year(x['year']), axis=1)
    # df['year_type'] = year_type
    # print(df['year_type'].value_counts())
    # print()
    #print(df.memory_usage(deep=True))
    #x = 3
    #print("runs value ... ", runs)
    #results_file = open("results.txt", "a")

    for run in range(runs):
        r = random.randint(0, run)
        df_mini = df.sample(n=10000,random_state=r) # fix random state
        year_type = df_mini.apply(lambda x: check_year(x['year']), axis=1)
        df_mini['year_type'] = year_type
        vals = df_mini['year_type'].value_counts()
        
        #results_file.write("run...{x}\n".format(x=run))
        #esults_file.close()
        print("run...{x}".format(x=run))
        #print()
        #print(vals)
        #print()
        #print(df_mini)

        #year_type = df.apply(lambda x: check_year(x['year']), axis=1)
        
        #print(df.head())
        #vals = df['year_type'].value_counts()
        #min_count = vals.min()

        #print(df['year_type'].value_counts())
        #print("Inspecting df of sample ...")
        #print(df.groupby('year_type').count())
        #print("Df shape", df.shape)
        #print(df.head())
        #print("DF columns: ", df.columns)
        #print("Duplicate rows: ")
        #print(df.duplicated())
        #print(" 5 papers with most citation count" )
        #print(df.nlargest(5, 'citation_count'))
        #print(" 5 papers with least citation count" )
        #print(df.nsmallest(5, 'citation_count'))
        #print("Unique")
        #print(df.nunique())

        #print("obtained df, going to train now...")
        train, test = train_test_split(df_mini, train_size=0.80)
        vect = TfidfVectorizer(stop_words='english', strip_accents='unicode', token_pattern=r"[a-zA-Z0-9']{2,}",\
            ngram_range=(1,2))
        X_tr = vect.fit_transform(train['abstract'])
        #print(terms)
        X_te = vect.transform(test['abstract'])

        #pca = PCA(n_components=2)   # PCA does not support sparse input
        svd = TruncatedSVD(n_components=n_components)
        data2D_tr = svd.fit_transform(X_tr)
        data2D_te = svd.transform(X_te)

        #model_path = input("Model path: ")
        model = KMeans(n_clusters=c)
        model.fit(data2D_tr)

        with PdfPages("silhouette_" + str(run) + ".pdf") as pdf:
            see_clusters(model, data2D_tr, pdf)

        # print("Components: ")
        # print(svd.components_)
        # print()

        # print("Variance Ratio: ")
        # print(svd.explained_variance_)
        # print()

        # print("Explained Variance Ratio")
        # print(svd.explained_variance_ratio_)
        # print()

        # print("Singular Values")
        # print(svd.singular_values_)
        # print()
        
        #pickle.dump(model, open(model_path, 'wb'))

        new_train = pd.DataFrame(columns=train.columns)
        new_train = pd.concat([train.reset_index(drop=True), pd.DataFrame(data2D_tr)], axis=1)
        new_train['cluster'] = model.labels_

        #print("X_tr shape: ", X_tr.shape)
        #print("X_te shape: ", X_te.shape)
        #print("data2D_tr shape: ", data2D_tr.shape)
        #print("data2D_te shape: ", data2D_te.shape)

        test_pred = model.predict(data2D_te)
        silhouette_avg = silhouette_score(X_te, test_pred)

        #results_file.write("Silhouette avg score: {x}\n".format(x=silhouette_avg))

        print("Silhouette avg score(s): ", silhouette_avg)
        print()

        centers2D = model.cluster_centers_
        order_centroids = centers2D.argsort()[:, ::-1]  

        terms = vect.get_feature_names()
        #print("SVD Components, ", svd.components_.shape)
        #print("first comp ", svd.components_[0].shape)
        #print("second com")

        #results_file.write("SVD INFO : ---------\n")
        print("SVD Info: -------")
        print()

        sorted_zip = sorted(zip(svd.components_[0], terms), key=lambda x: x[0])
        features_0 =  ["%+0.3f*%s" % (coef, feat) for coef, feat in sorted_zip]
        features_0_str =   " ".join(features_0)
        #features_0 = features_0.argsort()[::-1]
        print("SVD Comp 0: ", features_0[:10])
        #results_file.write("SVD Comp 0 features: ", features_0[:10], )

        sorted_zip = sorted(zip(svd.components_[1], terms), key=lambda x: x[0])
        features_1 =   ["%+0.3f*%s" % (coef, feat) for coef, feat in sorted_zip]
        features_1_str = " ".join(features_1)
        #features_1 = features_1.argsort()[::-1]
        print("SVD Comp 1: ", features_1[:10])

        sorted_zip = sorted(zip(svd.components_[2], terms), key=lambda x: x[0])
        features_2 =  ["%+0.3f*%s" % (coef, feat) for coef, feat in sorted_zip]
        features_2_str = " ".join(features_2)
        #features_2 = features_2.argsort()[::-1]
        print("SVD Comp 2: ", features_2[:10])
        
        print()
        print()
    
        print("Best features: --- ")
        best_features_0 = [terms[i] for i in svd.components_[0].argsort()[::-1]]
        print("SVD Comp 0: ", best_features_0[:10])
        print()

        best_features_1 = [terms[i] for i in svd.components_[1].argsort()[::-1]]
        print("SVD Comp 1: " , best_features_1[:10])
        print()

        best_features_2 = [terms[i] for i in svd.components_[2].argsort()[::-1]]
        print("SVD Comp 2: ", best_features_2[:10])
        print()

        # cluster_terms = []
        # for i in range(c):
        #     c_terms = " ".join([terms[ind] for ind in order_centroids[i, :c]])
        #     cluster_terms.append(c_terms)
        #     print("C_terms ", c_terms)
        #     #print("Cluster {x} terms: ".format(x=i), c_terms)
        # print(cluster_terms)

        comp_dict = {}
        for i in range(n_components):
            comp_dict[i] = "Component " + str(i)
        
        #print(svd.components_.shape)
        #print(df_mini.shape)
        #print(len(list(comp_dict.keys())))
        #print(pd.DataFrame(svd.components_,columns=df_mini.columns,index = list(comp_dict.keys())))

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
