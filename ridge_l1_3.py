# getting necessary libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

#sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
# getting the data
df1 = pd.read_csv('/flash/TerenzioU/DNA_l1_15_0.6.csv')
#df1.rename( columns={'Unnamed: 0':'site'}, inplace=True )
df1.columns = [ 'sites', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']
#print(df1)
#df1.['Unnamed: 1'] = (df1.['Unnamed: 1']).applymap(np.absolute)
#print(df1.['Unnamed: 1'])
#df = df1.T
#df = df.abs()
#plotting ridgeplot
#df = df.apply(pd.to_numeric) 
print('dtype =', df1.dtypes)
#print(df1)

for i in df1.columns:
  pal = sns.color_palette(palette='coolwarm', n_colors=12)
  g = sns.FacetGrid(df1, aspect = 6, height= 9, palette=pal)
  g.map(sns.kdeplot, i, bw_adjust=1, clip_on=False, fill=True, alpha=1, linewidth=1.5) 

'''
# we generate a color palette with Seaborn.color_palette()
pal = sns.color_palette(palette='coolwarm', n_colors=12)
# in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
g = sns.FacetGrid(df1, aspect=15, height=0.75, palette=pal)
n = np.arange(0,100)
# then we add the densities kdeplots for each sites
g.map(sns.kdeplot,'sites', bw_adjust=1, clip_on=False, fill=True, alpha=1, linewidth=1.5)
# here we add a white line that represents the contour of each kdeplot
g.map(sns.kdeplot,'sites', bw_adjust=1, clip_on=False, color="w", lw=2)
# here we add a horizontal line for each plot
g.map(plt.axhline, y=0,lw=2, clip_on=False)
# we loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
# notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
#for i, ax in enumerate(g.axes.flat):
    #ax.text(-15, 0.02, month_dict[i+1],fontweight='bold', fontsize=15,color=ax.lines[-1].get_color())
 # Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)
g.map(label, 'sites')
# we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
g.fig.subplots_adjust(hspace=-0.3)
# eventually we remove axes titles, yticks and spines
g.set_titles("Polaron")
g.set(yticks=[])
g.despine(bottom=True, left=True)
#plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
#g.map(label, "sites")
plt.xlabel('lattice sites', fontweight='bold', fontsize=15)
#g.fig.suptitle('Polaron',ha='right',fontsize=20,fontweight=20).    #plt.show()
'''
plt.savefig('/flash/TerenzioU/program/DNA_100_0.0_2.png')
