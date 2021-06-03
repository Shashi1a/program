import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cmath
import sympy as sp
import scipy.linalg as la
#from numpy import linalg as la


#parameters   --------------------------------------------------------------------
m=0.031; D=0.04; a=4.45; k=0.04; rho=0.5; beta=0.35; V=0.1; chi=0.6; gamma=0.005; hbar=0.00066
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
    H1 = (-V * (x[98] + x[0]) + chi * y[99] * x[99])/(hbar * sp.I)        #* (0+1j))
    #print('H1=', H1)
    '''H3 = -0.005 * z[99] + 11.4838709677419 * (-1.0 + sp.exp(-4.45 * y[99])) * sp.exp(-4.45 * y[99]) + 0.112903225806452 * (
            (-y[99] + y[1])**2) * sp.exp(-0.35 * y[99] - 0.35 * y[1]) + 0.112903225806452 * ((y[99] - y[98])**2) * sp.exp(
            -0.35 * y[99] - 0.35 * y[98]) - 32.258064516129 * (2 * y[99] - 2 * y[1]) * (
                         0.01 * sp.exp(-0.35 * y[99] - 0.35 * y[1]) + 0.02) - 32.258064516129 * (2 * y[99] - 2 * y[98]) * (
                         0.01 * sp.exp(-0.35 * y[99] - 0.35 * y[98]) + 0.02) - 19.3548387096774 * (np.abs(x[99])**2)
    H3 =  -m*(0.6 * (np.abs(x[99]))** 2 - 0.356 * (-1 + sp.exp(-4.45 * y[99])) * sp.exp(-4.45 * y[99]) + (-0.02 * y[0] + 0.02 * y[99]) * sp.exp(
        -0.35 * y[0] - 0.35 * y[99]) - 0.0035 * (y[0] - y[99]) ** 2 * sp.exp(-0.35 * y[0] - 0.35 * y[99]) - 0.0035 * (
                     -y[98] + y[99]) ** 2 * sp.exp(-0.35 * y[98] - 0.35 * y[99]) + (-0.02 * y[98] + 0.02 * y[99]) * sp.exp(
        -0.35 * y[98] - 0.35 * y[99])) - m * 0.005 * z[99]'''
    #H2 = np.float64(-0.005 * z[99]) + np.float64(11.4838709677419) * (np.float64(-1.0) + cmath.exp(np.float64(-4.45*y[99])) * cmath.exp(np.float64(-4.45*y[99])) + np.float64(0.112903225806452) * (np.float64(-y[99]) + (np.float64(y[0]))**2) * cmath.exp(np.float64(-0.35 * y[99])) - np.float64(0.35 *y[0])) + np.float64(0.112903225806452) * (np.float64(y[99])- (np.float64(y[98]))**2) * cmath.exp(-np.float64(0.35*y[99]) - np.float64(0.35) * np.float64(y[98])) - np.float64(32.258064516129) * (np.float64(2*y[99]) - np.float64(2*y[0])) * (np.float64(0.01) * cmath.exp(np.float64(-0.35) * np.float64(y[99]) - np.float64(0.35) * np.float64(y[0])) + np.float64(0.02)) - np.float64(32.258064516129) * (np.float64(2) * np.float64(y[99]) - np.float64(2) * np.float64(y[98])) * ( np.float64(0.01) * cmath.exp(np.float64(-0.35) * np.float64(y[99]) - np.float64(0.35) * np.float64(y[98])) + np.float64(0.02)) - np.float64(19.3548387096774) * np.float64(x[99]) * np.float64(x[99]) #* sp.conjugate(x[99])
    H3 = - 0.005 * z[99] + 11.4838709677419 * (-1 + sp.exp(-4.45 * y[99])) * sp.exp(-4.45 * y[99]) + 0.112903225806452 * (
                -y[99] + y[0]) ** 2 * sp.exp(-0.35 * y[99] - 0.35 * y[0]) + 0.112903225806452 * (
                     y[99] - y[98]) ** 2 * sp.exp(-0.35 * y[99] - 0.35 * y[98]) - 32.258064516129 * (
                     2 * y[99] - 2 * y[0]) * (
                     0.01 * sp.exp(-0.35 * y[99] - 0.35 * y[0]) + 0.02) - 32.258064516129 * (
                     2 * y[99] - 2 * y[98]) * (
                     0.01 * sp.exp(-0.35 * y[99] - 0.35 * y[98]) + 0.02) - 19.3548387096774 * (np.abs(x[99])**2)
    #H3 = -0.005 * z[99] - 19.3548387096774 * (np.abs(x[99])**2)
    H2 = z[99]
    E1.append(H1)
    E2.append(H2)
    E3.append(H3)
    eq = E1, E2, E3
    eq = np.reshape(eq, (300,))
    return eq


w1 = DNA(t,x,y,z)
w2 = sp.Matrix(w1)
w3 = [x, y, z]             #, E_0]
w3 = np.reshape(w3, (300,))

#w3 = np.append(w3, t)
w4 = w2.jacobian(w3)



#Initialisation  ----------------------------------------------------
x_0= np.zeros(100, dtype=np.complex128)
#x0[50] = 1

x_0[50] = 0.5058
x_0[49] ,x_0[51] = 0.45641589, 0.45641589
x_0[48] ,x_0[52] = 0.33535635, 0.33535635
x_0[47] ,x_0[53] = 0.20039376, 0.20063937
'''x0[46] ,x0[54] = 0.097743896, 0.097743896
x0[45] ,x0[55] = 0.038772784, 0.038772784
#x0[44] ,x0[56] = 0.012523571, 0.012523571
x0[47] = 0.0665+0j #0.00443185    #0.01842896        #, x0[54] = 0.1854, 0.1854
x0[48] = 0.2323+0j#0.05399097 #-0.03259717        #, x0[55] = 0.1073, 0.1073
x0[49] = 0.4919+0j#0.24197072
x0[50] = 0.6316+0j#0.39894228
x0[51] = 0.4919+0j #0.24197072     #0.01842896        #, x0[54] = 0.1854, 0.1854
x0[52] = 0.2323+0j #0.05399097 #-0.03259717        #, x0[55] = 0.1073, 0.1073
x0[53] = 0.0665+0j '''      #0.00443185     #-0.05920072         #, x0[56] = 0.0532, 0.0532
y_0 = np.zeros(100, dtype=np.complex128)
#y0[50] =  -0.345
#y0[50] =  -0.0969 #-0.3022061    #-1.3201#-0.047867
#y0[43] =  -0.00000410915 #-0.02284782#       0.03396376       #-1.320  #-0.091
#y0[44] =  -0.00005940493  #-0.09591296   #    -0.02959129
#y0[45] =  -0.000569403   #-0.22682804  #       0.09514149
#y0[46] =  -0.0036186403
'''y0[47] =  -0.015247 #-0.02284782#       0.03396376       #-1.320  #-0.091
y0[48] =  -0.042597  #-0.09591296   #    -0.02959129
y0[49] =  -0.0789019   #-0.22682804  #       0.09514149
y0[50] =  -0.0969 #-0.3022061    #-1.3201#-0.047867
y0[51] =  -0.0789019  #-0.22682804    #0.00715836
y0[52] =  -0.042597   #-0.09591296    #-0.03083426
y0[53] =  -0.015247  #-0.02284782   #-0.1267071'''
#y0[54] =  -0.0036186403 #-0.02284782#       0.03396376       #-1.320  #-0.091
#y0[55] =  -0.000569403  #-0.09591296   #    -0.02959129
#y0[56] =  -0.00005940493   #-0.22682804  #       0.09514149
#y0[57] =  -0.00000410915
y_0[47] =  -0.015247 #-0.02284782#       0.03396376       #-1.320  #-0.091
y_0[48] =  -0.042597  #-0.09591296   #    -0.02959129
y_0[49] =  -0.0789019   #-0.22682804  #       0.09514149
y_0[50] =  -0.3022061    #-1.3201#-0.047867
y_0[51] =  -0.0789019  #-0.22682804    #0.00715836
y_0[52] =  -0.042597   #-0.09591296    #-0.03083426
y_0[53] =  -0.015247  #-0.02284782   #-0.1267071

z_0 = np.zeros(100, dtype=np.complex128)    #s0 = [x0, y0, z0]; s0 = np.reshape(s0, (300,))
#p2 = eval('DNA(x0,y0,z0)')
t_0 = 0

values = {'t': t_0, 'x0': x_0[0], 'x1': x_0[1], 'x2': x_0[2], 'x3': x_0[3], 'x4': x_0[4], 'x5': x_0[5], 'x6': x_0[6], 'x7': x_0[7], 'x8': x_0[8], 'x9': x_0[9], 'x10': x_0[10],'x11': x_0[11],  'x12': x_0[12], 'x13': x_0[13], 'x14': x_0[14], 'x15': x_0[15], 'x16': x_0[16], 'x17': x_0[17], 'x18': x_0[18], 'x19': x_0[19], 'x20': x_0[20], 'x21': x_0[21], 'x22': x_0[22], 'x23': x_0[23], 'x24': x_0[24], 'x25': x_0[25], 'x26': x_0[26], 'x27': x_0[27], 'x28': x_0[28], 'x29': x_0[29], 'x30': x_0[30], 'x31': x_0[31], 'x32': x_0[32], 'x33': x_0[33], 'x34': x_0[34], 'x35': x_0[35], 'x36': x_0[36], 'x37': x_0[37], 'x38': x_0[38], 'x39': x_0[39], 'x40': x_0[40], 'x41': x_0[41], 'x42': x_0[42], 'x43': x_0[43], 'x44': x_0[44], 'x45': x_0[45], 'x46': x_0[46], 'x47': x_0[47], 'x48': x_0[48], 'x49': x_0[49], 'x50': x_0[50], 'x51': x_0[51], 'x52': x_0[52], 'x53': x_0[53], 'x54': x_0[54], 'x55': x_0[55], 'x56': x_0[56], 'x57': x_0[57], 'x58': x_0[58], 'x59': x_0[59], 'x60': x_0[60], 'x61': x_0[61], 'x62': x_0[62], 'x63': x_0[63], 'x64': x_0[64], 'x65': x_0[65], 'x66': x_0[66], 'x67': x_0[67], 'x68': x_0[68], 'x69': x_0[69], 'x70': x_0[70], 'x71': x_0[71], 'x72': x_0[72], 'x73': x_0[73], 'x74': x_0[74], 'x75': x_0[75], 'x76': x_0[76], 'x77': x_0[77], 'x78': x_0[78], 'x79': x_0[79],'x80': x_0[80], 'x81': x_0[81], 'x82': x_0[82], 'x83': x_0[83], 'x84': x_0[84], 'x85': x_0[85], 'x86': x_0[86], 'x87': x_0[87], 'x88': x_0[88], 'x89': x_0[89], 'x90': x_0[90], 'x91': x_0[91], 'x92': x_0[92], 'x93': x_0[93], 'x94': x_0[94], 'x95': x_0[95], 'x96': x_0[96], 'x97': x_0[97], 'x98': x_0[98], 'x99': x_0[99],'y0': y_0[0], 'y1': y_0[1], 'y2': y_0[2], 'y3': y_0[3], 'y4': y_0[4], 'y5': y_0[5], 'y6': y_0[6], 'y7': y_0[7], 'y8': y_0[8], 'y9': y_0[9], 'y10': y_0[10],'y11': y_0[11],  'y12': y_0[12], 'y13': y_0[13], 'y14': y_0[14], 'y15': y_0[15], 'y16': y_0[16], 'y17': y_0[17], 'y18': y_0[18], 'y19': y_0[19], 'y20': y_0[20], 'y21': y_0[21], 'y22': y_0[22], 'y23': y_0[23], 'y24': y_0[24], 'y25': y_0[25], 'y26': y_0[26], 'y27': y_0[27], 'y28': y_0[28], 'y29': y_0[29], 'y30': y_0[30], 'y31': y_0[31], 'y32': y_0[32], 'y33': y_0[33], 'y34': y_0[34], 'y35': y_0[35], 'y36': y_0[36], 'y37': y_0[37], 'y38': y_0[38], 'y39': y_0[39], 'y40': y_0[40], 'y41': y_0[41], 'y42': y_0[42], 'y43': y_0[43], 'y44': y_0[44], 'y45': y_0[45], 'y46': y_0[46], 'y47': y_0[47], 'y48': y_0[48], 'y49': y_0[49], 'y50': y_0[50], 'y51': y_0[51], 'y52': y_0[52], 'y53': y_0[53], 'y54': y_0[54], 'y55': y_0[55], 'y56': y_0[56], 'y57': y_0[57], 'y58': y_0[58], 'y59': y_0[59], 'y60': y_0[60], 'y61': y_0[61], 'y62': y_0[62], 'y63': y_0[63], 'y64': y_0[64], 'y65': y_0[65], 'y66': y_0[66], 'y67': y_0[67], 'y68': y_0[68], 'y69': y_0[69], 'y70': y_0[70], 'y71': y_0[71], 'y72': y_0[72], 'y73': y_0[73], 'y74': y_0[74], 'y75': y_0[75], 'y76': y_0[76], 'y77': y_0[77], 'y78': y_0[78], 'y79': y_0[79],'y80': y_0[80], 'y81': y_0[81], 'y82': y_0[82], 'y83': y_0[83], 'y84': y_0[84], 'y85': y_0[85], 'y86': y_0[86], 'y87': y_0[87], 'y88': y_0[88], 'y89': y_0[89], 'y90': y_0[90], 'y91': y_0[91], 'y92': y_0[92], 'y93': y_0[93], 'y94': y_0[94], 'y95': y_0[95], 'y96': y_0[96], 'y97': y_0[97], 'y98': y_0[98], 'y99': y_0[99]}
#w5= w4.subs(values)
#w5= sp.Matrix(w5, dtype=np.float64)
#w4 = sp.matrices.dense.matrix2numpy(w4, dtype=np.complex128)
print('shape of w4=', np.shape(w4))
#w4 = np.array(w4, dtype=np.complex128)
#print('w4=', w4)
#print('size of w4=', np.shape(w4))
#print(size of w4=', np.shape(w4))
w5 = w4.subs(values)
print('w5=', w5)
eigenv = la.eig(w5)
#w5 = eigenv.subs(values)
print('value =', w5)
print('eigen value =', eigenv)


l1 = np.zeros((1000,100), dtype=np.complex128);m1 = np.zeros((1000,100),dtype=np.complex128);n1 = np.zeros((1000,100),dtype=np.complex128)
l1[0, :] = x_0[0:100] #print('l1=', l1)
print('probability = ', sum((abs(x) **2) for x in l1[0, :]))
print('participation number =', 1/sum((abs(x) **4) for x in l1[0, :]))
m1[0, :] = y_0[0:100]
n1[0, :] = z_0[0:100]
h = 0.00001
t = np.zeros((1000))
t[0] = t_0
#eig1 = np.zeros(100, dtype=np.complex128); eig2 = np.zeros(100, dtype=np.complex128); eig3 = np.zeros(100, dtype=np.complex128); total_eig = np.zeros(100, dtype=np.complex128)


#RK4 method --------------------------------------------------------
for i in range(20):
    print('i=', i)
    #if sum(np.array(np.abs(l1)))<=1:
        #print('2=',i)
    #l1[i, :] = l1[i, :] + E1 * h
    #m1[i, :] = m1[i, :] + E1 * h
    #n1[i, :] = n1[i, :] + E1 * h
    a1 = DNA(t[i], l1[i, :], m1[i, :], n1[i, :])[0:100]
    #print('a1=', a1)
    b1 = DNA(t[i], (l1[i, :]), m1[i, :], n1[i, :])[100:200]
    #print('b1=', b1)
    c1 = DNA(t[i], (l1[i, :]), m1[i, :], n1[i, :])[200:300]
    #print('c1=', c1)
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
    print('3=', sum((np.abs(x) ** 2) for x in l1[i + 1, :]))
    sum_square = 1/sp.sqrt(sum(np.abs(x) ** 2 for x in l1[i+1, :]))
    print('sumsquare = ', sum_square)
    print('*****************************************************************************************************') 
    '''eig1[i] = la.eig(l1[i+1, :])
    eig2[i] = la.eig(m1[i+1, :])
    eig3[i] = la.eig(n1[i+1, :])
    total_eig[i] = eig1 + eig2 + eig3
    print('eig1 =', eig1[i]) 
    print('eig2 =', eig2[i]) 
    print('eig3 =', eig3[i]) 
    print('eigenvalue =', total_eig[i])'''
    #values = {'t': t_0, 'x0': l1[i, :], 'x1': x_0[1], 'x2': x_0[2], 'x3': x_0[3], 'x4': x_0[4], 'x5': x_0[5], 'x6': x_0[6], 'x7': x_0[7], 'x8': x_0[8], 'x9': x_0[9], 'x10': x_0[10],'x11': x_0[11],  'x12': x_0[12], 'x13': x_0[13], 'x14': x_0[14], 'x15': x_0[15], 'x16': x_0[16], 'x17': x_0[17], 'x18': x_0[18], 'x19': x_0[19], 'x20': x_0[20], 'x21': x_0[21], 'x22': x_0[22], 'x23': x_0[23], 'x24': x_0[24], 'x25': x_0[25], 'x26': x_0[26], 'x27': x_0[27], 'x28': x_0[28], 'x29': x_0[29], 'x30': x_0[30], 'x31': x_0[31], 'x32': x_0[32], 'x33': x_0[33], 'x34': x_0[34], 'x35': x_0[35], 'x36': x_0[36], 'x37': x_0[37], 'x38': x_0[38], 'x39': x_0[39], 'x40': x_0[40], 'x41': x_0[41], 'x42': x_0[42], 'x43': x_0[43], 'x44': x_0[44], 'x45': x_0[45], 'x46': x_0[46], 'x47': x_0[47], 'x48': x_0[48], 'x49': x_0[49], 'x50': x_0[50], 'x51': x_0[51], 'x52': x_0[52], 'x53': x_0[53], 'x54': x_0[54], 'x55': x_0[55], 'x56': x_0[56], 'x57': x_0[57], 'x58': x_0[58], 'x59': x_0[59], 'x60': x_0[60], 'x61': x_0[61], 'x62': x_0[62], 'x63': x_0[63], 'x64': x_0[64], 'x65': x_0[65], 'x66': x_0[66], 'x67': x_0[67], 'x68': x_0[68], 'x69': x_0[69], 'x70': x_0[70], 'x71': x_0[71], 'x72': x_0[72], 'x73': x_0[73], 'x74': x_0[74], 'x75': x_0[75], 'x76': x_0[76], 'x77': x_0[77], 'x78': x_0[78], 'x79': x_0[79],'x80': x_0[80], 'x81': x_0[81], 'x82': x_0[82], 'x83': x_0[83], 'x84': x_0[84], 'x85': x_0[85], 'x86': x_0[86], 'x87': x_0[87], 'x88': x_0[88], 'x89': x_0[89], 'x90': x_0[90], 'x91': x_0[91], 'x92': x_0[92], 'x93': x_0[93], 'x94': x_0[94], 'x95': x_0[95], 'x96': x_0[96], 'x97': x_0[97], 'x98': x_0[98], 'x99': x_0[99],'y0': y_0[0], 'y1': y_0[1], 'y2': y_0[2], 'y3': y_0[3], 'y4': y_0[4], 'y5': y_0[5], 'y6': y_0[6], 'y7': y_0[7], 'y8': y_0[8], 'y9': y_0[9], 'y10': y_0[10],'y11': y_0[11],  'y12': y_0[12], 'y13': y_0[13], 'y14': y_0[14], 'y15': y_0[15], 'y16': y_0[16], 'y17': y_0[17], 'y18': y_0[18], 'y19': y_0[19], 'y20': y_0[20], 'y21': y_0[21], 'y22': y_0[22], 'y23': y_0[23], 'y24': y_0[24], 'y25': y_0[25], 'y26': y_0[26], 'y27': y_0[27], 'y28': y_0[28], 'y29': y_0[29], 'y30': y_0[30], 'y31': y_0[31], 'y32': y_0[32], 'y33': y_0[33], 'y34': y_0[34], 'y35': y_0[35], 'y36': y_0[36], 'y37': y_0[37], 'y38': y_0[38], 'y39': y_0[39], 'y40': y_0[40], 'y41': y_0[41], 'y42': y_0[42], 'y43': y_0[43], 'y44': y_0[44], 'y45': y_0[45], 'y46': y_0[46], 'y47': y_0[47], 'y48': y_0[48], 'y49': y_0[49], 'y50': y_0[50], 'y51': y_0[51], 'y52': y_0[52], 'y53': y_0[53], 'y54': y_0[54], 'y55': y_0[55], 'y56': y_0[56], 'y57': y_0[57], 'y58': y_0[58], 'y59': y_0[59], 'y60': y_0[60], 'y61': y_0[61], 'y62': y_0[62], 'y63': y_0[63], 'y64': y_0[64], 'y65': y_0[65], 'y66': y_0[66], 'y67': y_0[67], 'y68': y_0[68], 'y69': y_0[69], 'y70': y_0[70], 'y71': y_0[71], 'y72': y_0[72], 'y73': y_0[73], 'y74': y_0[74], 'y75': y_0[75], 'y76': y_0[76], 'y77': y_0[77], 'y78': y_0[78], 'y79': y_0[79],'y80': y_0[80], 'y81': y_0[81], 'y82': y_0[82], 'y83': y_0[83], 'y84': y_0[84], 'y85': y_0[85], 'y86': y_0[86], 'y87': y_0[87], 'y88': y_0[88], 'y89': y_0[89], 'y90': y_0[90], 'y91': y_0[91], 'y92': y_0[92], 'y93': y_0[93], 'y94': y_0[94], 'y95': y_0[95], 'y96': y_0[96], 'y97': y_0[97], 'y98': y_0[98], 'y99': y_0[99]
    new_values = {'t': t[i], 'x0': l1[i, i, :], 'x1': l1[i, 1], 'x2': l1[i, 2], 'x3': l1[i, 3], 'x4': l1[i, 4], 'x5': l1[i, 5], 'x6': l1[i, 6],
          'x7': l1[i, 7], 'x8': l1[i, 8], 'x9': l1[i, 9], 'x10': l1[i, 10], 'x11': l1[i, 11], 'x12': l1[i, 12], 'x13': l1[i, 13],
          'x14': l1[i, 14], 'x15': l1[i, 15], 'x16': l1[i, 16], 'x17': l1[i, 17], 'x18': l1[i, 18], 'x19': l1[i, 19],
          'x20': l1[i, 20], 'x21': l1[i, 21], 'x22': l1[i, 22], 'x23': l1[i, 23], 'x24': l1[i, 24], 'x25': l1[i, 25],
          'x26': l1[i, 26], 'x27': l1[i, 27], 'x28': l1[i, 28], 'x29': l1[i, 29], 'x30': l1[i, 30], 'x31': l1[i, 31],
          'x32': l1[i, 32], 'x33': l1[i, 33], 'x34': l1[i, 34], 'x35': l1[i, 35], 'x36': l1[i, 36], 'x37': l1[i, 37],
          'x38': l1[i, 38], 'x39': l1[i, 39], 'x40': l1[i, 40], 'x41': l1[i, 41], 'x42': l1[i, 42], 'x43': l1[i, 43],
          'x44': l1[i, 44], 'x45': l1[i, 45], 'x46': l1[i, 46], 'x47': l1[i, 47], 'x48': l1[i, 48], 'x49': l1[i, 49],
          'x50': l1[i, 50], 'x51': l1[i, 51], 'x52': l1[i, 52], 'x53': l1[i, 53], 'x54': l1[i, 54], 'x55': l1[i, 55],
          'x56': l1[i, 56], 'x57': l1[i, 57], 'x58': l1[i, 58], 'x59': l1[i, 59], 'x60': l1[i, 60], 'x61': l1[i, 61],
          'x62': l1[i, 62], 'x63': l1[i, 63], 'x64': l1[i, 64], 'x65': l1[i, 65], 'x66': l1[i, 66], 'x67': l1[i, 67],
          'x68': l1[i, 68], 'x69': l1[i, 69], 'x70': l1[i, 70], 'x71': l1[i, 71], 'x72': l1[i, 72], 'x73': l1[i, 73],
          'x74': l1[i, 74], 'x75': l1[i, 75], 'x76': l1[i, 76], 'x77': l1[i, 77], 'x78': l1[i, 78], 'x79': l1[i, 79],
          'x80': l1[i, 80], 'x81': l1[i, 81], 'x82': l1[i, 82], 'x83': l1[i, 83], 'x84': l1[i, 84], 'x85': l1[i, 85],
          'x86': l1[i, 86], 'x87': l1[i, 87], 'x88': l1[i, 88], 'x89': l1[i, 89], 'x90': l1[i, 90], 'x91': l1[i, 91],
          'x92': l1[i, 92], 'x93': l1[i, 93], 'x94': l1[i, 94], 'x95': l1[i, 95], 'x96': l1[i, 96], 'x97': l1[i, 97],
          'x98': l1[i, 98], 'x99': l1[i, 99], 'y0': m1[i, 0], 'y1': m1[i, 1], 'y2': m1[i, 2], 'y3': m1[i, 3], 'y4': m1[i, 4],
          'y5': m1[i, 5], 'y6': m1[i, 6], 'y7': m1[i, 7], 'y8': m1[i, 8], 'y9': m1[i, 9], 'y10': m1[i, 10], 'y11': m1[i, 11],
          'y12': m1[i, 12], 'y13': m1[i, 13], 'y14': m1[i, 14], 'y15': m1[i, 15], 'y16': m1[i, 16], 'y17': m1[i, 17],
          'y18': m1[i, 18], 'y19': m1[i, 19], 'y20': m1[i, 20], 'y21': m1[i, 21], 'y22': m1[i, 22], 'y23': m1[i, 23],
          'y24': m1[i, 24], 'y25': m1[i, 25], 'y26': m1[i, 26], 'y27': m1[i, 27], 'y28': m1[i, 28], 'y29': m1[i, 29],
          'y30': m1[i, 30], 'y31': m1[i, 31], 'y32': m1[i, 32], 'y33': m1[i, 33], 'y34': m1[i, 34], 'y35': m1[i, 35],
          'y36': m1[i, 36], 'y37': m1[i, 37], 'y38': m1[i, 38], 'y39': m1[i, 39], 'y40': m1[i, 40], 'y41': m1[i, 41],
          'y42': m1[i, 42], 'y43': m1[i, 43], 'y44': m1[i, 44], 'y45': m1[i, 45], 'y46': m1[i, 46], 'y47': m1[i, 47],
          'y48': m1[i, 48], 'y49': m1[i, 49], 'y50': m1[i, 50], 'y51': m1[i, 51], 'y52': m1[i, 52], 'y53': m1[i, 53],
          'y54': m1[i, 54], 'y55': m1[i, 55], 'y56': m1[i, 56], 'y57': m1[i, 57], 'y58': m1[i, 58], 'y59': m1[i, 59],
          'y60': m1[i, 60], 'y61': m1[i, 61], 'y62': m1[i, 62], 'y63': m1[i, 63], 'y64': m1[i, 64], 'y65': m1[i, 65],
          'y66': m1[i, 66], 'y67': m1[i, 67], 'y68': m1[i, 68], 'y69': m1[i, 69], 'y70': m1[i, 70], 'y71': m1[i, 71],
          'y72': m1[i, 72], 'y73': m1[i, 73], 'y74': m1[i, 74], 'y75': m1[i, 75], 'y76': m1[i, 76], 'y77': m1[i, 77],
          'y78': m1[i, 78], 'y79': m1[i, 79], 'y80': m1[i, 80], 'y81': m1[i, 81], 'y82': m1[i, 82], 'y83': m1[i, 83],
          'y84': m1[i, 84], 'y85': m1[i, 85], 'y86': m1[i, 86], 'y87': m1[i, 87], 'y88': m1[i, 88], 'y89': m1[i, 89],
          'y90': m1[i, 90], 'y91': m1[i, 91], 'y92': m1[i, 92], 'y93': m1[i, 93], 'y94': m1[i, 94], 'y95': m1[i, 95],
          'y96': m1[i, 96], 'y97': m1[i, 97], 'y98': m1[i, 98], 'y99': m1[i, 99],
          'z0': l1[i, i, :], 'z1': n1[i, 1], 'z2': n1[i, 2], 'z3': n1[i, 3], 'z4': n1[i, 4], 'z5': n1[i, 5], 'z6': n1[i, 6],
          'z7': n1[i, 7], 'z8': n1[i, 8], 'z9': n1[i, 9], 'z10': n1[i, 10], 'z11': n1[i, 11], 'z12': n1[i, 12], 'z13': n1[i, 13],
          'z14': n1[i, 14], 'z15': n1[i, 15], 'z16': n1[i, 16], 'z17': n1[i, 17], 'z18': n1[i, 18], 'z19': n1[i, 19],
          'z20': n1[i, 20], 'z21': n1[i, 21], 'z22': n1[i, 22], 'z23': n1[i, 23], 'z24': n1[i, 24], 'z25': n1[i, 25],
          'z26': n1[i, 26], 'z27': n1[i, 27], 'z28': n1[i, 28], 'z29': n1[i, 29], 'z30': n1[i, 30], 'z31': n1[i, 31],
          'z32': n1[i, 32], 'z33': n1[i, 33], 'z34': n1[i, 34], 'z35': n1[i, 35], 'z36': n1[i, 36], 'z37': n1[i, 37],
          'z38': n1[i, 38], 'z39': n1[i, 39], 'z40': n1[i, 40], 'z41': n1[i, 41], 'z42': n1[i, 42], 'z43': n1[i, 43],
          'z44': n1[i, 44], 'z45': n1[i, 45], 'z46': n1[i, 46], 'z47': n1[i, 47], 'z48': n1[i, 48], 'z49': n1[i, 49],
          'z50': n1[i, 50], 'z51': n1[i, 51], 'z52': n1[i, 52], 'z53': n1[i, 53], 'z54': n1[i, 54], 'z55': n1[i, 55],
          'z56': n1[i, 56], 'z57': n1[i, 57], 'z58': n1[i, 58], 'z59': n1[i, 59], 'z60': n1[i, 60], 'z61': n1[i, 61],
          'z62': n1[i, 62], 'z63': n1[i, 63], 'z64': n1[i, 64], 'z65': n1[i, 65], 'z66': n1[i, 66], 'z67': n1[i, 67],
          'z68': n1[i, 68], 'z69': n1[i, 69], 'z70': n1[i, 70], 'z71': n1[i, 71], 'z72': n1[i, 72], 'z73': n1[i, 73],
          'z74': n1[i, 74], 'z75': n1[i, 75], 'z76': n1[i, 76], 'z77': n1[i, 77], 'z78': n1[i, 78], 'z79': n1[i, 79],
          'z80': n1[i, 80], 'z81': n1[i, 81], 'z82': n1[i, 82], 'z83': n1[i, 83], 'z84': n1[i, 84], 'z85': n1[i, 85],
          'z86': n1[i, 86], 'z87': n1[i, 87], 'z88': n1[i, 88], 'z89': n1[i, 89], 'z90': n1[i, 90], 'z91': n1[i, 91],
          'z92': n1[i, 92], 'z93': n1[i, 93], 'z94': n1[i, 94], 'z95': n1[i, 95], 'z96': n1[i, 96], 'z97': n1[i, 97],
          'z98': n1[i, 98], 'z99': n1[i, 99]}
    #w5= w4.subs(new_values)
    w6 =  w4.subs(new_value)
    print('vvalue =', w6)
    eigen = la.eig(w6)
    print('new eigen value [',i,'] =', eigen)
    print('****************************************************') 
    #print('l ki value are=', np.abs(l1[i+1,:]))
    #print('m ki value are=', m1[i+1,:])
    #print('n ki value are=', n1[i+1,:])


'''
        #print('3=',sum(np.array(np.abs(l1))))
    plt.plot(np.linspace(0, 100, 100), np.abs(l1[i+1, :]))
    plt.plot(np.linspace(0, 100, 100), m1[i+1, :])
    plt.plot(np.linspace(0, 100, 100), n1[i+1, :])
    plt.ylim(-0.1, 1)
plt.show()

'''
