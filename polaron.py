import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

df = pd.read_csv('/flash/TerenzioU/DNA_l1_15_0.6.csv')
df.head()



for i in df.columns:
  pal = sns.color_palette(palette='coolwarm', n_colors=12)
  g = sns.FacetGrid(df, aspect = 6, height= 9, palette=pal)
  g.map(sns.kdeplot, i, bw_adjust=1, clip_on=False, fill=True, alpha=1, linewidth=1.5) 
  
  
plt.savefig('/flash/TerenzioU/program/pola.png')
