import numpy as np
from mpl_toolkits.mplot3d import Axes 3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import cmath
import sympy as sp
import scipy.linalg as la
#from numpy import linalg as la
from ipywidgets import interactive




#parameters   --------------------------------------------------------------------
m=0.031; D=0.04; a=4.45; k=0.04; rho=0.5; beta=0.35; V=0.1; chi=0.0; gamma=0.005; hbar=0.00066
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


'''w1 = DNA(t,x,y,z)
w2 = sp.Matrix(w1)
w3 = [x, y, z]             #, E_0]
w3 = np.reshape(w3, (9,))

w3 = np.append(w3, t)
w4 = w2.jacobian(w3)'''



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
y_0 = np.zeros(15, dtype=np.complex128)
#y_0[1] = -0.3022
#y0[50] =  -0.345
#y0[50] =  -0.0969 #-0.3022061    #-1.3201#-0.047867
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
'''y_0[4] =  -0.015247 #-0.02284782#       0.03396376       #-1.320  #-0.091
y_0[5] =  -0.042597  #-0.09591296   #    -0.02959129
y_0[6] =  -0.0789019   #-0.22682804  #       0.09514149
y_0[7] =  -0.3022061    #-1.3201#-0.047867
y_0[8] =  -0.0789019  #-0.22682804    #0.00715836
y_0[9] =  -0.042597   #-0.09591296    #-0.03083426
y_0[10] =  -0.015247  #-0.02284782   #-0.1267071
'''
z_0 = np.zeros(15, dtype=np.complex128)    #s0 = [x0, y0, z0]; s0 = np.reshape(s0, (300,))
#p2 = eval('DNA(x0,y0,z0)')
t_0 = 0
#values = {'t': t_0, 'x0': x_0[0], 'x1': x_0[1], 'x2': x_0[2], 'y0': y_0[0], 'y1': y_0[1], 'y2': y_0[2], 'z0': z_0[0], 'z1': z_0[1], 'z2': z_0[2]}
#values = {'t': t_0, 'x0': x_0[0], 'x1': x_0[1], 'x2': x_0[2], 'x3': x_0[3], 'x4': x_0[4], 'x5': x_0[5], 'x6': x_0[6], 'x7': x_0[7], 'x8': x_0[8], 'x9': x_0[9], 'x10': x_0[10],'x11': x_0[11],  'x12': x_0[12], 'x13': x_0[13], 'x14': x_0[14], 'x15': x_0[15], 'x16': x_0[16], 'x17': x_0[17], 'x18': x_0[18], 'x19': x_0[19], 'x20': x_0[20], 'x21': x_0[21], 'x22': x_0[22], 'x23': x_0[23], 'x24': x_0[24], 'x25': x_0[25], 'x26': x_0[26], 'x27': x_0[27], 'x28': x_0[28], 'x29': x_0[29], 'x30': x_0[30], 'x31': x_0[31], 'x32': x_0[32], 'x33': x_0[33], 'x34': x_0[34], 'x35': x_0[35], 'x36': x_0[36], 'x37': x_0[37], 'x38': x_0[38], 'x39': x_0[39], 'x40': x_0[40], 'x41': x_0[41], 'x42': x_0[42], 'x43': x_0[43], 'x44': x_0[44], 'x45': x_0[45], 'x46': x_0[46], 'x47': x_0[47], 'x48': x_0[48], 'x49': x_0[49], 'x50': x_0[50], 'x51': x_0[51], 'x52': x_0[52], 'x53': x_0[53], 'x54': x_0[54], 'x55': x_0[55], 'x56': x_0[56], 'x57': x_0[57], 'x58': x_0[58], 'x59': x_0[59], 'x60': x_0[60], 'x61': x_0[61], 'x62': x_0[62], 'x63': x_0[63], 'x64': x_0[64], 'x65': x_0[65], 'x66': x_0[66], 'x67': x_0[67], 'x68': x_0[68], 'x69': x_0[69], 'x70': x_0[70], 'x71': x_0[71], 'x72': x_0[72], 'x73': x_0[73], 'x74': x_0[74], 'x75': x_0[75], 'x76': x_0[76], 'x77': x_0[77], 'x78': x_0[78], 'x79': x_0[79],'x80': x_0[80], 'x81': x_0[81], 'x82': x_0[82], 'x83': x_0[83], 'x84': x_0[84], 'x85': x_0[85], 'x86': x_0[86], 'x87': x_0[87], 'x88': x_0[88], 'x89': x_0[89], 'x90': x_0[90], 'x91': x_0[91], 'x92': x_0[92], 'x93': x_0[93], 'x94': x_0[94], 'x95': x_0[95], 'x96': x_0[96], 'x97': x_0[97], 'x98': x_0[98], 'x99': x_0[99],'y0': y_0[0], 'y1': y_0[1], 'y2': y_0[2], 'y3': y_0[3], 'y4': y_0[4], 'y5': y_0[5], 'y6': y_0[6], 'y7': y_0[7], 'y8': y_0[8], 'y9': y_0[9], 'y10': y_0[10],'y11': y_0[11],  'y12': y_0[12], 'y13': y_0[13], 'y14': y_0[14], 'y15': y_0[15], 'y16': y_0[16], 'y17': y_0[17], 'y18': y_0[18], 'y19': y_0[19], 'y20': y_0[20], 'y21': y_0[21], 'y22': y_0[22], 'y23': y_0[23], 'y24': y_0[24], 'y25': y_0[25], 'y26': y_0[26], 'y27': y_0[27], 'y28': y_0[28], 'y29': y_0[29], 'y30': y_0[30], 'y31': y_0[31], 'y32': y_0[32], 'y33': y_0[33], 'y34': y_0[34], 'y35': y_0[35], 'y36': y_0[36], 'y37': y_0[37], 'y38': y_0[38], 'y39': y_0[39], 'y40': y_0[40], 'y41': y_0[41], 'y42': y_0[42], 'y43': y_0[43], 'y44': y_0[44], 'y45': y_0[45], 'y46': y_0[46], 'y47': y_0[47], 'y48': y_0[48], 'y49': y_0[49], 'y50': y_0[50], 'y51': y_0[51], 'y52': y_0[52], 'y53': y_0[53], 'y54': y_0[54], 'y55': y_0[55], 'y56': y_0[56], 'y57': y_0[57], 'y58': y_0[58], 'y59': y_0[59], 'y60': y_0[60], 'y61': y_0[61], 'y62': y_0[62], 'y63': y_0[63], 'y64': y_0[64], 'y65': y_0[65], 'y66': y_0[66], 'y67': y_0[67], 'y68': y_0[68], 'y69': y_0[69], 'y70': y_0[70], 'y71': y_0[71], 'y72': y_0[72], 'y73': y_0[73], 'y74': y_0[74], 'y75': y_0[75], 'y76': y_0[76], 'y77': y_0[77], 'y78': y_0[78], 'y79': y_0[79],'y80': y_0[80], 'y81': y_0[81], 'y82': y_0[82], 'y83': y_0[83], 'y84': y_0[84], 'y85': y_0[85], 'y86': y_0[86], 'y87': y_0[87], 'y88': y_0[88], 'y89': y_0[89], 'y90': y_0[90], 'y91': y_0[91], 'y92': y_0[92], 'y93': y_0[93], 'y94': y_0[94], 'y95': y_0[95], 'y96': y_0[96], 'y97': y_0[97], 'y98': y_0[98], 'y99': y_0[99]}
#w5= w4.subs(values)
#w5= sp.Matrix(w5, dtype=np.float64)
#w4 = sp.matrices.dense.matrix2numpy(w4, dtype=np.complex128)
#print('shape of w4=', np.shape(w4))
#w4 = np.array(w4, dtype=np.complex128)
#print('w4=', w4)
#print('size of w4=', np.shape(w4))
#print(size of w4=', np.shape(w4))
#w5 = w4.subs(values)
#print('w5=', w5)
#eigenv = la.eig(w5)
#w5 = eigenv.subs(values)
#print('value =', w5)
#print('eigen value =', eigenv)


l1 = np.zeros((1000,15), dtype=np.complex128);m1 = np.zeros((1000,15),dtype=np.complex128);n1 = np.zeros((1000,15),dtype=np.complex128)
l1[0, :] = x_0[0:15] #print('l1=', l1)
print('probability = ', sum((abs(x) **2) for x in l1[0, :]))
print('participation number =', 1/sum((abs(x) **4) for x in l1[0, :]))
m1[0, :] = y_0[0:15]
n1[0, :] = z_0[0:15]
h = 0.00001
t = np.zeros((1000))
t[0] = t_0
#eig1 = np.zeros(100, dtype=np.complex128); eig2 = np.zeros(100, dtype=np.complex128); eig3 = np.zeros(100, dtype=np.complex128); total_eig = np.zeros(100, dtype=np.complex128)


#RK4 method --------------------------------------------------------
for i in range(998):
    print('i=', i)
    #if sum(np.array(np.abs(l1)))<=1:
        #print('2=',i)
    #l1[i, :] = l1[i, :] + E1 * h
    #m1[i, :] = m1[i, :] + E1 * h
    #n1[i, :] = n1[i, :] + E1 * h
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
    #print('*****************************************************************************************************') 
    '''eig1[i] = la.eig(l1[i+1, :])
    eig2[i] = la.eig(m1[i+1, :])
    eig3[i] = la.eig(n1[i+1, :])
    total_eig[i] = eig1 + eig2 + eig3
    print('eig1 =', eig1[i]) 
    print('eig2 =', eig2[i]) 
    print('eig3 =', eig3[i]) 
    print('eigenvalue =', total_eig[i])'''
    #values = {'t': t_0, 'x0': l1[i, :], 'x1': x_0[1], 'x2': x_0[2], 'x3': x_0[3], 'x4': x_0[4], 'x5': x_0[5], 'x6': x_0[6], 'x7': x_0[7], 'x8': x_0[8], 'x9': x_0[9], 'x10': x_0[10],'x11': x_0[11],  'x12': x_0[12], 'x13': x_0[13], 'x14': x_0[14], 'x15': x_0[15], 'x16': x_0[16], 'x17': x_0[17], 'x18': x_0[18], 'x19': x_0[19], 'x20': x_0[20], 'x21': x_0[21], 'x22': x_0[22], 'x23': x_0[23], 'x24': x_0[24], 'x25': x_0[25], 'x26': x_0[26], 'x27': x_0[27], 'x28': x_0[28], 'x29': x_0[29], 'x30': x_0[30], 'x31': x_0[31], 'x32': x_0[32], 'x33': x_0[33], 'x34': x_0[34], 'x35': x_0[35], 'x36': x_0[36], 'x37': x_0[37], 'x38': x_0[38], 'x39': x_0[39], 'x40': x_0[40], 'x41': x_0[41], 'x42': x_0[42], 'x43': x_0[43], 'x44': x_0[44], 'x45': x_0[45], 'x46': x_0[46], 'x47': x_0[47], 'x48': x_0[48], 'x49': x_0[49], 'x50': x_0[50], 'x51': x_0[51], 'x52': x_0[52], 'x53': x_0[53], 'x54': x_0[54], 'x55': x_0[55], 'x56': x_0[56], 'x57': x_0[57], 'x58': x_0[58], 'x59': x_0[59], 'x60': x_0[60], 'x61': x_0[61], 'x62': x_0[62], 'x63': x_0[63], 'x64': x_0[64], 'x65': x_0[65], 'x66': x_0[66], 'x67': x_0[67], 'x68': x_0[68], 'x69': x_0[69], 'x70': x_0[70], 'x71': x_0[71], 'x72': x_0[72], 'x73': x_0[73], 'x74': x_0[74], 'x75': x_0[75], 'x76': x_0[76], 'x77': x_0[77], 'x78': x_0[78], 'x79': x_0[79],'x80': x_0[80], 'x81': x_0[81], 'x82': x_0[82], 'x83': x_0[83], 'x84': x_0[84], 'x85': x_0[85], 'x86': x_0[86], 'x87': x_0[87], 'x88': x_0[88], 'x89': x_0[89], 'x90': x_0[90], 'x91': x_0[91], 'x92': x_0[92], 'x93': x_0[93], 'x94': x_0[94], 'x95': x_0[95], 'x96': x_0[96], 'x97': x_0[97], 'x98': x_0[98], 'x99': x_0[99],'y0': y_0[0], 'y1': y_0[1], 'y2': y_0[2], 'y3': y_0[3], 'y4': y_0[4], 'y5': y_0[5], 'y6': y_0[6], 'y7': y_0[7], 'y8': y_0[8], 'y9': y_0[9], 'y10': y_0[10],'y11': y_0[11],  'y12': y_0[12], 'y13': y_0[13], 'y14': y_0[14], 'y15': y_0[15], 'y16': y_0[16], 'y17': y_0[17], 'y18': y_0[18], 'y19': y_0[19], 'y20': y_0[20], 'y21': y_0[21], 'y22': y_0[22], 'y23': y_0[23], 'y24': y_0[24], 'y25': y_0[25], 'y26': y_0[26], 'y27': y_0[27], 'y28': y_0[28], 'y29': y_0[29], 'y30': y_0[30], 'y31': y_0[31], 'y32': y_0[32], 'y33': y_0[33], 'y34': y_0[34], 'y35': y_0[35], 'y36': y_0[36], 'y37': y_0[37], 'y38': y_0[38], 'y39': y_0[39], 'y40': y_0[40], 'y41': y_0[41], 'y42': y_0[42], 'y43': y_0[43], 'y44': y_0[44], 'y45': y_0[45], 'y46': y_0[46], 'y47': y_0[47], 'y48': y_0[48], 'y49': y_0[49], 'y50': y_0[50], 'y51': y_0[51], 'y52': y_0[52], 'y53': y_0[53], 'y54': y_0[54], 'y55': y_0[55], 'y56': y_0[56], 'y57': y_0[57], 'y58': y_0[58], 'y59': y_0[59], 'y60': y_0[60], 'y61': y_0[61], 'y62': y_0[62], 'y63': y_0[63], 'y64': y_0[64], 'y65': y_0[65], 'y66': y_0[66], 'y67': y_0[67], 'y68': y_0[68], 'y69': y_0[69], 'y70': y_0[70], 'y71': y_0[71], 'y72': y_0[72], 'y73': y_0[73], 'y74': y_0[74], 'y75': y_0[75], 'y76': y_0[76], 'y77': y_0[77], 'y78': y_0[78], 'y79': y_0[79],'y80': y_0[80], 'y81': y_0[81], 'y82': y_0[82], 'y83': y_0[83], 'y84': y_0[84], 'y85': y_0[85], 'y86': y_0[86], 'y87': y_0[87], 'y88': y_0[88], 'y89': y_0[89], 'y90': y_0[90], 'y91': y_0[91], 'y92': y_0[92], 'y93': y_0[93], 'y94': y_0[94], 'y95': y_0[95], 'y96': y_0[96], 'y97': y_0[97], 'y98': y_0[98], 'y99': y_0[99]
    #new_values = {'t': t[i], 'x0': l1[i, i, :], 'x1': l1[i, 1], 'x2': l1[i, 2], 'y0': m1[i, 0], 'y1': m1[i, 1], 'y2': m1[i, 2], 'z0': l1[i, i, :], 'z1': n1[i, 1], 'z2': n1[i, 2]}
    #w5= w4.subs(new_values)
    #w6 =  w4.subs(new_value)
    #print('vvalue =', w6)
    #eigen = la.eig(w6)
    #print('new eigen value [',i,'] =', eigen)
    #print('****************************************************') 
    #print('l ki value are=', np.abs(l1[i+1,:]))
    #print('m ki value are=', m1[i+1,:])
    #print('n ki value are=', n1[i+1,:])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Data for a three-dimensional line
    zline = np.linspace(0.00001, .01, 997)
    xline = np.linspace(0, 100, 100)
    yline = np.linspace(0, 1, 100)
    #ax.plot3D(xline, yline, zline, 'gray')
    
    # Data for three-dimensional scattered points
    z = np.abs(l1[i+1, :])
    x = np.arange(0,15)
    y = np.arange(0.00001, .001)
    #ax.plot3D(xdata, ydata, zdata, 'green')
    ax.plot_surface(x, y, z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.view_init(eliv = E, azim = A)
    ax.set_title('Polaron n = 15')
    ax.set_xlim(0, 15); ax.set_zlim(-0.1, 1)
    iplot = interactive(plotter, E = (-90,90,5), A = (-90,90,5))

     	
    #plt.plot(np.linspace(0, 100, 100), np.abs(l1[i+1, :]))
    #plt.plot(np.linspace(0, 100, 100), m1[i+1, :])
    #plt.plot(np.linspace(0, 100, 100), n1[i+1, :])
    #plt.ylim(-0.1, 1)
'''fig, ax = plt.subplots()
     #print('3=',sum(np.array(np.abs(l1))))
plt.plot(np.linspace(0, 15, 15), np.abs(l1[997, :]))
plt.plot(np.linspace(0, 15, 15), m1[997, :])
#plt.plot(np.linspace(0, 15, 15), n1[i, :])
plt.ylim(-0.4, 1)'''
plt.savefig('/flash/TerenzioU/program/DNAgraph_1/0.0.png')
iplot

