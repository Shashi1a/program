import pandas as pd
import joypy
import numpy as np
import matplotlib.pyplot as plt
#set some display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#import the csv
df2 = pd.read_csv('/flash/TerenzioU/DNA_l1_15_0.6.csv')
#create a color gradent function to be used in the colormap parameter
def color_gradient(x=0.0, start=(0, 0, 0), stop=(1, 1, 1)):
    r = np.interp(x, [0, 1], [start[0], stop[0]])
    g = np.interp(x, [0, 1], [start[1], stop[1]])
    b = np.interp(x, [0, 1], [start[2], stop[2]])
    return (r, g, b)
#show the table
print(df2.head(3))
#plot the figure
plt.figure(dpi=380)
fig, axes = joypy.joyplot(df2
                          , overlap=2.5
                          , by="index"
                          , ylim='own'
                          , x_range=(0,60)
                          , fill=True
                          , figsize=(10,13)
                          , legend=False
                          , xlabels=True
                          , ylabels=True
                          #, color=['#76a5af', '#134f5c']
                          , colormap=lambda x: color_gradient(x, start=(.08, .45, .8)
                                                             ,stop=(.8, .34, .44))
                          , alpha=0.6
                          , linewidth=.5
                          , linecolor='w'
                          #, background='k' # change to 'k' for black background or 'grey' for grey
                          , fade=True)
plt.title('Joy Plot of Polaron'
          , fontsize=14
          , color='blue'
          , alpha=1)
plt.rc("font", size=12)
plt.xlabel('sites', fontsize=14, color='blue', alpha=1)
plt.ylabel('time', fontsize=8, color='blue', alpha=1)
plt.savefig('/flash/TerenzioU/program/DNA_15_0.6_1.png')
