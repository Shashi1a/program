import numpy as np
#from mpl_toolkits import mplot3d
#from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib import cm
import cmath
import sympy as sp
import scipy.linalg as la
#from numpy import linalg as la
import pandas as pd


#parameters   --------------------------------------------------------------------
m=0.031; D=0.04; a=4.45; k=0.04; rho=0.5; beta=0.35; V=0.1
chi=0.6
gamma=0.005; hbar=0.00066
n = 100
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24,x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47,x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70,x71, x72, x73, x74, x75, x76, x77, x78, x79, x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93,x94, x95, x96, x97, x98, x99,y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24,y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47,y48, y49, y50, y51, y52, y53, y54, y55, y56, y57, y58, y59, y60, y61, y62, y63, y64, y65, y66, y67, y68, y69, y70,y71, y72, y73, y74, y75, y76, y77, y78, y79, y80, y81, y82, y83, y84, y85, y86, y87, y88, y89, y90, y91, y92, y93,y94, y95, y96, y97, y98, y99,z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15, z16, z17, z18, z19, z20, z21, z22, z23, z24,z25, z26, z27, z28, z29, z30, z31, z32, z33, z34, z35, z36, z37, z38, z39, z40, z41, z42, z43, z44, z45, z46, z47,z48, z49, z50, z51, z52, z53, z54, z55, z56, z57, z58, z59, z60, z61, z62, z63, z64, z65, z66, z67, z68, z69, z70,z71, z72, z73, z74, z75, z76, z77, z78, z79, z80, z81, z82, z83, z84, z85, z86, z87, z88, z89, z90, z91, z92, z93,z94, z95, z96, z97, z98, z99 = sp.var('x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 x65 x66 x67 x68 x69 x70 x71 x72 x73 x74 x75 x76 x77 x78 x79 x80 x81 x82 x83 x84 x85 x86 x87 x88 x89 x90 x91 x92 x93 x94 x95 x96 x97 x98 x99 y0 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 y65 y66 y67 y68 y69 y70 y71 y72 y73 y74 y75 y76 y77 y78 y79 y80 y81 y82 y83 y84 y85 y86 y87 y88 y89 y90 y91 y92 y93 y94 y95 y96 y97 y98 y99 z0 z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14 z15 z16 z17 z18 z19 z20 z21 z22 z23 z24 z25 z26 z27 z28 z29 z30 z31 z32 z33 z34 z35 z36 z37 z38 z39 z40 z41 z42 z43 z44 z45 z46 z47 z48 z49 z50 z51 z52 z53 z54 z55 z56 z57 z58 z59 z60 z61 z62 z63 z64 z65 z66 z67 z68 z69 z70 z71 z72 z73 z74 z75 z76 z77 z78 z79 z80 z81 z82 z83 z84 z85 z86 z87 z88 z89 z90 z91 z92 z93 z94 z95 z96 z97 z98 z99')
x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24,x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47,x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70,x71, x72, x73, x74, x75, x76, x77, x78, x79, x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93,x94, x95, x96, x97, x98, x99]
y = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24,y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47,y48, y49, y50, y51, y52, y53, y54, y55, y56, y57, y58, y59, y60, y61, y62, y63, y64, y65, y66, y67, y68, y69, y70,y71, y72, y73, y74, y75, y76, y77, y78, y79, y80, y81, y82, y83, y84, y85, y86, y87, y88, y89, y90, y91, y92, y93,y94, y95, y96, y97, y98, y99]
z = [z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15, z16, z17, z18, z19, z20, z21, z22, z23, z24,z25, z26, z27, z28, z29, z30, z31, z32, z33, z34, z35, z36, z37, z38, z39, z40, z41, z42, z43, z44, z45, z46, z47,z48, z49, z50, z51, z52, z53, z54, z55, z56, z57, z58, z59, z60, z61, z62, z63, z64, z65, z66, z67, z68, z69, z70,z71, z72, z73, z74, z75, z76, z77, z78, z79, z80, z81, z82, z83, z84, z85, z86, z87, z88, z89, z90, z91, z92, z93,z94, z95, z96, z97, z98, z99]
t = sp.var('t')

#defining equation as a function -----------------------------------------------------
def DNA(t,x,y,z):
    E1=[]; E2=[]; E3=[]
    for n in range(99):
        I1 = (-V * (x[n + 1] + x[n - 1]) + chi * y[n] * x[n])/(hbar * sp.I)
	I3 = - 0.005 * z[n] + 11.4838709677419 * (-1.0 + sp.exp(-4.45 * y[n])) * sp.exp(-4.45 * y[n]) + 0.112903225806452 * (
		(-y[n] + y[n+1])**2) * sp.exp(-0.35 * y[n] - 0.35 * y[n+1]) + 0.112903225806452 * ((y[n] - y[n-1])**2) * sp.exp(
		-0.35 * y[n] - 0.35 * y[n-1]) - 32.258064516129 * (2 * y[n] - 2 * y[n+1]) * (
		0.01 * sp.exp(-0.35 * y[n] - 0.35 * y[n+1]) + 0.02) - 32.258064516129 * (2 * y[n] - 2 * y[n-1]) * (
		0.01 * sp.exp(-0.35 * y[n] - 0.35 * y[n-1]) + 0.02) - 19.3548387096774 * (np.abs(x[99])**2)
	I2 = z[n]
	E1.append(I1)
	E2.append(I2)
	E3.append(I3)
    H1 = (-V * (x[98] + x[0]) + chi * y[99] * x[99])/(hbar * sp.I)        #* (0+1j)
    #print('H1=', H1)
    H3 = -0.005 * z[99] + 11.4838709677419 * (-1.0 + sp.exp(-4.45 * y[99])) * sp.exp(-4.45 * y[99]) + 0.112903225806452 * (
	    (-y[99] + y[0])**2) * sp.exp(-0.35 * y[99] - 0.35 * y[0]) + 0.112903225806452 * ((y[99] - y[98])**2) * sp.exp(
	    -0.35 * y[99] - 0.35 * y[98]) - 32.258064516129 * (2 * y[99] - 2 * y[0]) * (
	    0.01 * sp.exp(-0.35 * y[99] - 0.35 * y[0]) + 0.02) - 32.258064516129 * (2 * y[99] - 2 * y[98]) * (
	    0.01 * sp.exp(-0.35 * y[99] - 0.35 * y[98]) + 0.02) - 19.3548387096774 * (np.abs(x[99])**2)
    H2 = z[99]
    E1.append(H1)
    E2.append(H2)
    E3.append(H3)
    eq = E1, E2, E3
    eq = np.reshape(eq, (300,))
    return eq

#Initialisation  ----------------------------------------------------
x_0= np.zeros(100, dtype=np.complex128)
#x0[50] = 1
x_0[50] = 0.5058
x_0[49] ,x_0[51] = 0.45641589, 0.45641589
x_0[48] ,x_0[52] = 0.33535635, 0.33535635
x_0[47] ,x_0[53] = 0.20039376, 0.20063937

y_0 = np.zeros(100, dtype=np.complex128)
#y0[43] =  -0.00000410915 #-0.02284782#       0.03396376       #-1.320  #-0.091
#y0[44] =  -0.00005940493  #-0.09591296   #    -0.02959129
#y0[45] =  -0.000569403   #-0.22682804  #       0.09514149
#y0[46] =  -0.0036186403
y_0[47] =  -0.015247 #-0.02284782#       0.03396376       #-1.320  #-0.091
y_0[48] =  -0.042597  #-0.09591296   #    -0.02959129
y_0[49] =  -0.0789019   #-0.22682804  #       0.09514149
y_0[50] =  -0.0969 #-0.3022061    #-1.3201#-0.047867
y_0[51] =  -0.0789019  #-0.22682804    #0.00715836
y_0[52] =  -0.042597   #-0.09591296    #-0.03083426
y_0[53] =  -0.015247  #-0.02284782   #-0.1267071'''
#y0[54] =  -0.0036186403 #-0.02284782#       0.03396376       #-1.320  #-0.091
#y0[55] =  -0.000569403  #-0.09591296   #    -0.02959129
#y0[56] =  -0.00005940493   #-0.22682804  #       0.09514149
#y0[57] =  -0.00000410915

z_0 = np.zeros(100, dtype=np.complex128)    #s0 = [x0, y0, z0]; s0 = np.reshape(s0, (300,))
t_0 = 0

l1 = np.zeros((1000,100), dtype=np.complex128);m1 = np.zeros((1000,100),dtype=np.complex128);n1 = np.zeros((1000,100),dtype=np.complex128)
l1[0, :] = x_0[0:100] #print('l1=', l1)
print('probability = ', sum((abs(x) **2) for x in l1[0, :]))
print('participation number =', 1/sum((abs(x) **4) for x in l1[0, :]))
m1[0, :] = y_0[0:100]
n1[0, :] = z_0[0:100]
h = 0.00001
t = np.zeros((1000))
t[0] = t_0

#RK4 method --------------------------------------------------------
for i in range(1000):
    print('i=', i)
    a1 = DNA(t[i], l1[i, :], m1[i, :], n1[i, :])[0:100]                      #print('a1=', a1)
    b1 = DNA(t[i], (l1[i, :]), m1[i, :], n1[i, :])[100:200]                  #print('b1=', b1)
    c1 = DNA(t[i], (l1[i, :]), m1[i, :], n1[i, :])[200:300]                    #print('c1=', c1)
	
    a2 = DNA(t[i] + 0.5 * h, (l1[i, :]) + 0.5 * a1 * h, m1[i, :] + 0.5 * b1 * h, n1[i, :] + 0.5 * c1 * h)[0:100]
    b2 = DNA(t[i] + 0.5 * h, (l1[i, :]) + 0.5 * a1 * h, m1[i, :] + 0.5 * b1 * h, n1[i, :] + 0.5 * c1 * h)[100:200]
    c2 = DNA(t[i] + 0.5 * h, (l1[i, :]) + 0.5 * a1 * h, m1[i, :] + 0.5 * b1 * h, n1[i, :] + 0.5 * c1 * h)[200:300]

    a3 = DNA(t[i] + 0.5 * h, (l1[i, :]) + 0.5 * a2 * h, m1[i, :] + 0.5 * b2 * h, n1[i, :] + 0.5 * c2 * h)[0:100]
    b3 = DNA(t[i] + 0.5 * h, (l1[i, :]) + 0.5 * a2 * h, m1[i, :] + 0.5 * b2 * h, n1[i, :] + 0.5 * c2 * h)[100:200]
    c3 = DNA(t[i] + 0.5 * h, (l1[i, :]) + 0.5 * a2 * h, m1[i, :] + 0.5 * b2 * h, n1[i, :] + 0.5 * c2 * h)[200:300]
	
    a4 = DNA(t[i] + h, (l1[i, :]) + a3 * h, m1[i, :] + b3 * h, n1[i, :] + c3 * h)[0:100]
    b4 = DNA(t[i] + h, (l1[i, :]) + a3 * h, m1[i, :] + b3 * h, n1[i, :] + c3 * h)[100:200]
    c4 = DNA(t[i] + h, (l1[i, :]) + a3 * h, m1[i, :] + b3 * h, n1[i, :] + c3 * h)[200:300]

    t[i+1] = t[i] + h
    l1[i+1, :] = l1[i, :] + (a1 + 2 * a2 + 2 * a3 + a4) * h / 6
    m1[i+1, :] = m1[i, :] + (b1 + 2 * b2 + 2 * b3 + b4) * h / 6
    n1[i+1, :] = n1[i, :] + (c1 + 2 * c2 + 2 * c3 + c4) * h / 6
    '''print('3=', sum((np.abs(x) ** 2) for x in l1[i + 1, :]))
    sum_square = 1/sp.sqrt(sum(np.abs(x) ** 2 for x in l1[i+1, :]))
    print('sumsquare = ', sum_square)
    print('*****************************************************************************************************') 
    #print('l ki value are=', np.abs(l1[i+1,:]))
    #print('m ki value are=', m1[i+1,:])
    #print('n ki value are=', n1[i+1,:])'''
#saving data in database and plotting 
    '''fig = plt.figure()
    ax = plt.gca(projection='3d')
    # Data for a three-dimensional line
    zline = np.linspace(0.00001, .01, 997)
    xline = np.linspace(0, 100, 100)
    yline = np.linspace(0, 1, 100)
    #ax.plot3D(xline, yline, zline, 'gray')
    
    # Data for three-dimensional scattered points
    z = t[i]
    x = i
    y = np.abs(l1[i, :])
    #ax.plot3D(xdata, ydata, zdata, 'green')
    ax.view_init(elev=45., azim=150)
    x, y = np.meshgrid(x, y, indexing='ij')
    ax.plot_surface(x, y, z, rstride=1, cstride=1,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title('Polaron')
    ax.set_xlim(0, 100); ax.set_ylim(-0.1, 1)
    ax.set_xlabel('n'); ax.set_ylabel('|x_{n}|'); ax.set_zlabel('t')'''
w1 = pd.DataFrame(l1).to_csv('/flash/TerenzioU/program/l1(100/0.6).csv')
w2 = pd.DataFrame(m1).to_csv('/flash/TerenzioU/program/m1(100/0.6).csv')
w3 = pd.DataFrame(n1).to_csv('/flash/TerenzioU/program/n1(100/0.6).csv')
fig, ax = plt.subplots()
plt.plot(np.linspace(0, 1, 100), np.abs(l1[998, :]))
plt.plot(np.linspace(0, 100, 100), m1[998, :])
#plt.plot(np.linspace(0, 100, 100), n1[i+1, :])
plt.ylim(-0.4, 1)
plt.savefig('/flash/TerenzioU/program/DNA_polaron_chi0.6.png')

