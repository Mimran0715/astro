import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.ion()

import random
import seaborn as sns 
import os

sns.set()

def main():
    path = '/Users/Mal/Documents/results'
    with PdfPages("results.pdf") as pdf: 
        for filename in os.listdir(path):
            if "test_pred" in filename:
                name = os.path.join(path, filename)
                f = plt.figure()
                ax = sns.histplot(data=pd.read_csv(name), x='year', hue='pred_str')
                ax.set_title("File: " + name + " test prediction results")
                plt.show()
                pdf.savefig(f)
    
        #if filename.contains("test_pred"):
        

if __name__ == '__main__':
    main()