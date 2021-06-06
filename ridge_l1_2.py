
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pandas as pd
import joypy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#set some display options
df1 = pd.read_csv('/flash/TerenzioU/DNA_l1_15_0.6.csv')
#create a color gradent function to be used in the colormap parameter
def color_gradient(x=0.0, start=(0, 0, 0), stop=(1, 1, 1)):
    r = np.interp(x, [0, 1], [start[0], stop[0]])
    g = np.interp(x, [0, 1], [start[1], stop[1]])
    b = np.interp(x, [0, 1], [start[2], stop[2]])
    return (r, g, b)
#show the table
#print(df2.head(3))
#plot the figure#3
#plt.figure()
plt.figure(figsize=(16,10), dpi= 80)
##%matplotlib inline

fig, axes = plt.subplots(1,1,1)
axes[0] = joypy.joyplot(df1)
#ax = plt.gca()
#plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.title('Polaron wave function'
          , fontsize=6
          , color='green')
plt.rc("font", size=6)
plt.xlabel('sites', fontsize=6, color='blue')
plt.ylabel('time', fontsize=6, color='blue')
plt.savefig('/flash/TerenzioU/program/DNA_100_0.0_2.png')
