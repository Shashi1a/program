import pandas as pd
import joypy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#set some display options
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
plt.figure()
fig, axes = joypy.joyplot(df2)
plt.title('Joy Plot of Polaron'
          , fontsize=8
          , color='green')
plt.rc("font", size=12)
plt.xlabel('sites', fontsize=8, color='blue')
plt.ylabel('time', fontsize=8, color='blue')
plt.savefig('/flash/TerenzioU/program/DNA_15_0.6_2.png')
