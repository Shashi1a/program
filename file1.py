import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import cmath
import sympy as sp
import scipy.linalg as la
#from numpy import linalg as la
#from ipywidgets import interactive
#import seaborn as sns
import joypy

#parameters   --------------------------------------------------------------------
m=0.031; D=0.04; a=4.45; k=0.04; rho=0.5; beta=0.35; V=0.1; chi=0.6; gamma=0.005; hbar=0.00066
n = 15
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15 = sp.var('x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 y0 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 z0 z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14 z15')
x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15]
y = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15]
z = [z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15]
t = sp.var('t')
#defining equation as a function -----------------------------------------------------
def DNA(t,x,y,z):
    E1=[]; E2=[]; E3=[]
    for n in range(14):
        I1 = (-V * (x[n + 1] + x[n - 1]) + chi * y[n] * x[n])/(hbar * sp.I)
        #I3 =  -0.005*z[n] - 19.3548387096774*(np.abs(x[n])**2)
 #+ 11.4838709677419*(-1 + sp.exp(-4.45*y[n]))*sp.exp(-4.45*y[n]) + 0.112903225806452*(-y[n] + y[n+1])**2*sp.exp(-0.35*y[n] - 0.35*y[n+1]) + 0.112903225806452*(y[n] - y[n-1])**2*sp.exp(-0.35*y[n] - 0.35*y[n-1]) - 32.258064516129*(2*y[n] - 2*y[n+1])*(0.01*sp.exp(-0.35*y[n] - 0.35*y[n+1]) + 0.02) - 32.258064516129*(2*y[n] - 2*y[n-1])*(0.01*sp.exp(-0.35*y[n] - 0.35*y[n-1]) + 0.02)
        I3 = - 0.005 * z[n] + 11.4838709677419 * (-1.0 + sp.exp(-4.45 * y[n])) * sp.exp(-4.45 * y[n]) + 0.112903225806452 * (
                (-y[n] + y[n+1])**2) * sp.exp(-0.35 * y[n] - 0.35 * y[n+1]) + 0.112903225806452 * ((y[n] - y[n-1])**2) * sp.exp(
            -0.35 * y[n] - 0.35 * y[n-1]) - 32.258064516129 * (2 * y[n] - 2 * y[n+1]) * (
                         0.01 * sp.exp(-0.35 * y[n] - 0.35 * y[n+1]) + 0.02) - 32.258064516129 * (2 * y[n] - 2 * y[n-1]) * (
                         0.01 * sp.exp(-0.35 * y[n] - 0.35 * y[n-1]) + 0.02) - 19.3548387096774 * (np.abs(x[n])**2)
	#I3 =  -m*(0.6 * (np.abs(x[n]))** 2 - 0.356 * (-1 + sp.exp(-4.45 * y[n])) * sp.exp(-4.45 * y[n]) + (-0.02 * y[n+1] + 0.02 * y[n]) * sp.exp( -0.35 * y[n+1] - 0.35 * y[n]) - 0.0035 * (y[n+1] - y[n]) ** 2 * sp.exp(-0.35 * y[n+1] - 0.35 * y[n]) - 0.0035 * (-y[n-1] + y[n]) ** 2 * sp.exp(-0.35 * y[n-1] - 0.35 * y[n]) + (-0.02 * y[n-1] + 0.02 * y[n]) * sp.exp(-0.35 * y[n-1] - 0.35 * y[n])) - m * 0.005 * z[n]
        #I2 = np.float64(-0.005 * z[n]) + np.float64(11.4838709677419) * (np.float64(-1.0) + cmath.exp(np.float64(-4.45*y[n])) * cmath.exp(np.float64(-4.45*y[n])) + np.float64(0.112903225806452) * (np.float64(-y[n]) + (np.float64(y[n+1]))**2) * cmath.exp(np.float64(-0.35 * y[n])) - np.float64(0.35*y[n+1])) + np.float64(0.112903225806452) * (np.float64(y[n])- (np.float64(y[n-1]))**2) * cmath.exp(-np.float64(0.35 * y[n]) - np.float64(0.35) * np.float64(y[n-1])) - np.float64(32.258064516129) * (np.float64(2*y[n]) - np.float64(2*y[n+1])) * (np.float64(0.01) * cmath.exp(np.float64(-0.35) * np.float64(y[n]) - np.float64(0.35) * np.float64(y[n+1])) + np.float64(0.02)) - np.float64(32.258064516129) * (np.float64(2) * np.float64(y[n]) - np.float64(2) * np.float64(y[n-1])) * (np.float64(0.01) * cmath.exp(np.float64(-0.35) * np.float64(y[n]) - np.float64(0.35) * np.float64(y[n-1])) + np.float64(0.02)) - np.float64(19.3548387096774) * np.float64(x[n]) * np.float64(x[n]) #* sp.conjugate(x[n])
        #print('I2=', I2)
#I3 =  -m*(0.6 * (np.abs(x[n])) ** 2 - 0.356 * (-1 + sp.exp(-4.45 * y[n])) * sp.exp(-4.45 * y[n]) + (-0.02 * y[n + 1] + 0.02 * y[n]) * sp.exp(-0.35 * y[n + 1] - 0.35 * y[n]) - 0.0035 * (y[n + 1] - y[n]) ** 2 * sp.exp( -0.35 * y[n + 1] - 0.35 * y[n]) - 0.0035 * ( -y[n - 1] + y[n]) ** 2 * sp.exp(-0.35 * y[n - 1] - 0.35 * y[n]) + (-0.02 * y[n - 1] + 0.02 * y[n]) * sp.exp(-0.35 * y[n - 1] - 0.35 * y[n])) - m*0.005*z[n]
       # I3 = -m*(0.6 * (np.abs(x[n])) ** 2 - 0.356 * (-1 + sp.exp(-4.45 * y[n])) * sp.exp(-4.45 * y[n]) + (-0.02 * y[n + 1] + 0.02 * y[n]) * sp.exp(-0.35 * y[n + 1] - 0.35 * y[n]) - 0.0035 * (y[n + 1] - y[n]) ** 2 * sp.exp(-0.35 * y[n + 1] - 0.35 * y[n]) - 0.0035 * (-y[n - 1] + y[n]) ** 2 * sp.exp(-0.35 * y[n - 1] - 0.35 * y[n]) + (-0.02 * y[n - 1] + 0.02 * y[n]) * sp.exp(-0.35 * y[n - 1] - 0.35 * y[n])) - m * 0.005 *z[n]
        I2 = z[n]
        E1.append(I1)
        E2.append(I2)
        E3.append(I3)
    H1 = (-V * (x[1] + x[0]) + chi * y[2] * x[2])/(hbar * sp.I)        #* (0+1j))
    #print('H1=', H1)
    H3 = -0.005 * z[14] + 11.4838709677419 * (-1.0 + sp.exp(-4.45 * y[14])) * sp.exp(-4.45 * y[14]) + 0.112903225806452 * (
            (-y[14] + y[0])**2) * sp.exp(-0.35 * y[14] - 0.35 * y[0]) + 0.112903225806452 * ((y[14] - y[13])**2) * sp.exp(
            -0.35 * y[14] - 0.35 * y[13]) - 32.258064516129 * (2 * y[14] - 2 * y[0]) * (
                         0.01 * sp.exp(-0.35 * y[14] - 0.35 * y[0]) + 0.02) - 32.258064516129 * (2 * y[14] - 2 * y[13]) * (
                         0.01 * sp.exp(-0.35 * y[14] - 0.35 * y[13]) + 0.02) - 19.3548387096774 * (np.abs(x[14])**2)
    '''H3 =  -m*(0.6 * (np.abs(x[99]))** 2 - 0.356 * (-1 + sp.exp(-4.45 * y[99])) * sp.exp(-4.45 * y[99]) + (-0.02 * y[0] + 0.02 * y[99]) * sp.exp(
        -0.35 * y[0] - 0.35 * y[99]) - 0.0035 * (y[0] - y[99]) ** 2 * sp.exp(-0.35 * y[0] - 0.35 * y[99]) - 0.0035 * (
                     -y[98] + y[99]) ** 2 * sp.exp(-0.35 * y[98] - 0.35 * y[99]) + (-0.02 * y[98] + 0.02 * y[99]) * sp.exp(
        -0.35 * y[98] - 0.35 * y[99])) - m * 0.005 * z[99]'''
    #H2 = np.float64(-0.005 * z[99]) + np.float64(11.4838709677419) * (np.float64(-1.0) + cmath.exp(np.float64(-4.45*y[99])) * cmath.exp(np.float64(-4.45*y[99])) + np.float64(0.112903225806452) * (np.float64(-y[99]) + (np.float64(y[0]))**2) * cmath.exp(np.float64(-0.35 * y[99])) - np.float64(0.35 *y[0])) + np.float64(0.112903225806452) * (np.float64(y[99])- (np.float64(y[98]))**2) * cmath.exp(-np.float64(0.35*y[99]) - np.float64(0.35) * np.float64(y[98])) - np.float64(32.258064516129) * (np.float64(2*y[99]) - np.float64(2*y[0])) * (np.float64(0.01) * cmath.exp(np.float64(-0.35) * np.float64(y[99]) - np.float64(0.35) * np.float64(y[0])) + np.float64(0.02)) - np.float64(32.258064516129) * (np.float64(2) * np.float64(y[99]) - np.float64(2) * np.float64(y[98])) * ( np.float64(0.01) * cmath.exp(np.float64(-0.35) * np.float64(y[99]) - np.float64(0.35) * np.float64(y[98])) + np.float64(0.02)) - np.float64(19.3548387096774) * np.float64(x[99]) * np.float64(x[99]) #* sp.conjugate(x[99])
    '''H3 = - 0.005 * z[99] + 11.4838709677419 * (-1 + sp.exp(-4.45 * y[99])) * sp.exp(-4.45 * y[99]) + 0.112903225806452 * (
                -y[99] + y[0]) ** 2 * sp.exp(-0.35 * y[99] - 0.35 * y[0]) + 0.112903225806452 * (
                     y[99] - y[98]) ** 2 * sp.exp(-0.35 * y[99] - 0.35 * y[98]) - 32.258064516129 * (
                     2 * y[99] - 2 * y[0]) * (
                     0.01 * sp.exp(-0.35 * y[99] - 0.35 * y[0]) + 0.02) - 32.258064516129 * (
                     2 * y[99] - 2 * y[98]) * (
                     0.01 * sp.exp(-0.35 * y[99] - 0.35 * y[98]) + 0.02) - 19.3548387096774 * (np.abs(x[99])**2)'''
    #H3 = -0.005 * z[99] - 19.3548387096774 * (np.abs(x[99])**2)
    H2 = z[14]
    E1.append(H1)
    E2.append(H2)
    E3.append(H3)
    eq = E1, E2, E3
    eq = np.reshape(eq, (45,))
    #print('eq=', eq)
    return eq


#Initialisation  ----------------------------------------------------
x_0= np.zeros(15, dtype=np.complex128)
#x0[50] = 1
#x_0[1] = 1
x_0[7] = 1      #0.5058
#x_0[6] ,x_0[8] = 0.45641589, 0.45641589
#x_0[5] ,x_0[9] = 0.33535635, 0.33535635
#x_0[4] ,x_0[10] = 0.20069376, 0.20063937
#x_0[3] ,x_0[11] = 0.097743896, 0.097743896
#x_0[2] ,x_0[12] = 0.038772784, 0.038772784
y_0 = np.zeros(15, dtype=np.complex128)
#y0[43] =  -0.00000410915 #-0.02284782#       0.03396376       #-1.320  #-0.091
#y0[44] =  -0.00005940493  #-0.09591296   #    -0.02959129
#y_0[2] =  -0.000569403   #-0.22682804  #       0.09514149
#y_0[3] =  -0.0036186403
#y_0[4] =  -0.015247 #-0.02284782#       0.03396376       #-1.320  #-0.091
#y_0[5] =  -0.042597  #-0.09591296   #    -0.02959129
#y_0[6] =  -0.0789019   #-0.22682804  #       0.09514149
#y_0[7] =  -0.0969 #-0.3022061    #-1.3201#-0.047867
#y_0[8] =  -0.0789019  #-0.22682804    #0.00715836
#y_0[9] =  -0.042597   #-0.09591296    #-0.03083426
#y_0[10] =  -0.015247  #-0.02284782   #-0.1267071
#y_0[11] =  -0.0036186403 #-0.02284782#       0.03396376       #-1.320  #-0.091
#y_0[12] =  -0.000569403  #-0.09591296   #    -0.02959129
#y0[56] =  -0.00005940493   #-0.22682804  #       0.09514149
#y0[57] =  -0.00000410915
z_0 = np.zeros(15, dtype=np.complex128)    #s0 = [x0, y0, z0]; s0 = np.reshape(s0, (300,))
t_0 = 0

#defining empty array -----------------------------------------------------------
l1 = np.zeros((1000,15), dtype=np.complex128);m1 = np.zeros((1000,15),dtype=np.complex128);n1 = np.zeros((1000,15),dtype=np.complex128)
l1[0, :] = x_0[0:15] #print('l1=', l1)
print('probability = ', sum((abs(x) **2) for x in l1[0, :]))
print('participation number =', 1/sum((abs(x) **4) for x in l1[0, :]))
m1[0, :] = y_0[0:15]
n1[0, :] = z_0[0:15]
h = 0.00001
t = np.zeros((1000))
t[0] = t_0

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_title('Polaron')
    
#RK4 method --------------------------------------------------------
for i in range(100):
    print('i=', i)

    a1 = DNA(t[i], l1[i, :], m1[i, :], n1[i, :])[0:15]
    #print('a1=', a1)
    b1 = DNA(t[i], (l1[i, :]), m1[i, :], n1[i, :])[15:30]
    #print('b1=', b1)
    c1 = DNA(t[i], (l1[i, :]), m1[i, :], n1[i, :])[30:45]
    #print('c1=', c1)
    a2 = DNA(t[i] + 0.5 * h, (l1[i, :]) + 0.5 * a1 * h, m1[i, :] + 0.5 * b1 * h, n1[i, :] + 0.5 * c1 * h)[0:15]
    b2 = DNA(t[i] + 0.5 * h, (l1[i, :]) + 0.5 * a1 * h, m1[i, :] + 0.5 * b1 * h, n1[i, :] + 0.5 * c1 * h)[15:30]
    c2 = DNA(t[i] + 0.5 * h, (l1[i, :]) + 0.5 * a1 * h, m1[i, :] + 0.5 * b1 * h, n1[i, :] + 0.5 * c1 * h)[30:45]
    a3 = DNA(t[i] + 0.5 * h, (l1[i, :]) + 0.5 * a2 * h, m1[i, :] + 0.5 * b2 * h, n1[i, :] + 0.5 * c2 * h)[0:15]
    b3 = DNA(t[i] + 0.5 * h, (l1[i, :]) + 0.5 * a2 * h, m1[i, :] + 0.5 * b2 * h, n1[i, :] + 0.5 * c2 * h)[15:30]
    c3 = DNA(t[i] + 0.5 * h, (l1[i, :]) + 0.5 * a2 * h, m1[i, :] + 0.5 * b2 * h, n1[i, :] + 0.5 * c2 * h)[30:45]
    a4 = DNA(t[i] + h, (l1[i, :]) + a3 * h, m1[i, :] + b3 * h, n1[i, :] + c3 * h)[0:15]
    b4 = DNA(t[i] + h, (l1[i, :]) + a3 * h, m1[i, :] + b3 * h, n1[i, :] + c3 * h)[15:30]
    c4 = DNA(t[i] + h, (l1[i, :]) + a3 * h, m1[i, :] + b3 * h, n1[i, :] + c3 * h)[30:45]
    t[i+1] = t[i] + h
    l1[i+1, :] = l1[i, :] + (a1 + 2 * a2 + 2 * a3 + a4) * h / 6
    m1[i+1, :] = m1[i, :] + (b1 + 2 * b2 + 2 * b3 + b4) * h / 6
    n1[i+1, :] = n1[i, :] + (c1 + 2 * c2 + 2 * c3 + c4) * h / 6
    #print('3=', sum((np.abs(x) ** 2) for x in l1[i + 1, :]))
    #sum_square = 1/sp.sqrt(sum(np.abs(x) ** 2 for x in l1[i+1, :]))
    #print('sumsquare = ', sum_square)
    #print('l ki value are=', np.abs(l1[i+1,:]))
    #print('m ki value are=', m1[i+1,:])
    #print('n ki value are=', n1[i+1,:])
    # Data for three-dimensional scattered points
    #def f(x, y):
      #return np.sin(np.sqrt(x ** 2 + y ** 2))

    x = np.arange(0,15)
    #z = np.arange(0,t[i])
    y = np.abs(l1[i, :])

    X, Y = np.meshgrid(x,y)
    #Z = f(X,Y)
    Z = X*np.exp(-X**2 - Y**2)
    #ax.set_xlim(0, 15); ax.set_zlim(-0.2, 1)
    
    ax.plot_wireframe(X, Y, Z, color='blue')
    #X, Y1 = np.meshgrid(x,y1)
    #ax.plot_wireframe(X, Y1, Z, color= 'red')
    ax.invert_xaxis()
    ax.view_init(20, -120)
    #ax.contour3D(X, Y, Z, 50, cmap='binary')
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
#w1 = np.column_stack([l1,m1,n1])
w1 = pd.DataFrame(np.abs(l1)).to_csv('./DNA_l1_15_0.6.csv')
w2 = pd.DataFrame(np.abs(m1)).to_csv('./DNA_m1_15_0.6.csv')
#plt.show()
plt.savefig('/flash/TerenzioU/program/polaron_15_2_0.6.png')
#vdf.append(w1)
#fdf = pd.concat(vdf).to_csv('/flash/TerenzioU/program/im_sp_data1.csv', sep=',', index=False, header=True)

#iplot = interactive(plotter, E = (-90,90,5), A = (-90,90,5))
#iplot

