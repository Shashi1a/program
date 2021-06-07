import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import scipy.optimize
from scipy.sparse.linalg import inv

D=0.04; a=4.45; k=0.04; rho=0.5; beta=0.35; V=0.1; chi=0.6
#n = 100
E = sp.var('E')
n = 15
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14 = sp.var('x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 y0 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 ')
x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]
y = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14]
#t = sp.var('t')
#x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24,x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47,x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70,x71, x72, x73, x74, x75, x76, x77, x78, x79, x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93,x94, x95, x96, x97, x98, x99,y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24,y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47, y48, y49, y50, y51, y52, y53, y54, y55, y56, y57, y58, y59, y60, y61, y62, y63, y64, y65, y66, y67, y68, y69, y70,y71, y72, y73, y74, y75, y76, y77, y78, y79, y80, y81, y82, y83, y84, y85, y86, y87, y88, y89, y90, y91, y92, y93,y94, y95, y96, y97, y98, y99 = sp.var('x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 x65 x66 x67 x68 x69 x70 x71 x72 x73 x74 x75 x76 x77 x78 x79 x80 x81 x82 x83 x84 x85 x86 x87 x88 x89 x90 x91 x92 x93 x94 x95 x96 x97 x98 x99 y0 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 y65 y66 y67 y68 y69 y70 y71 y72 y73 y74 y75 y76 y77 y78 y79 y80 y81 y82 y83 y84 y85 y86 y87 y88 y89 y90 y91 y92 y93 y94 y95 y96 y97 y98 y99')
#x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47,x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70,x71, x72, x73, x74, x75, x76, x77, x78, x79, x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93,x94, x95, x96, x97, x98, x99]
#y = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47,y48, y49, y50, y51, y52, y53, y54, y55, y56, y57, y58, y59, y60, y61, y62, y63, y64, y65, y66, y67, y68, y69, y70,y71, y72, y73, y74, y75, y76, y77, y78, y79, y80, y81, y82, y83, y84, y85, y86, y87, y88, y89, y90, y91, y92, y93,y94, y95, y96, y97, y98, y99]
# initialisation
x_0 = np.zeros(15, dtype=np.float64)
x_0[7] = 1
y_0 = np.zeros(15, dtype=np.float64)
y_0[7] =  -0.9545 #-0.3022061    #-1.3201#-0.047867
E_0 = -0.5727
s0 = [x_0, y_0]#, E_0]
s0 = np.reshape(s0, (30,))
s0 = np.append(s0, E_0)
s0 = s0.flatten()

s1 = [x, y]#, E_0]
s1 = np.reshape(s1, (30,))
s1 = np.append(s1, E)
#s2 = sp.Matrix(s1)
s1 = s1.flatten()

def F(x,y,E):
    E1 = [];E2 = [];E3 = []; E4=[]
    for n in range(99):
        I1 = V * (x[n + 1] + x[n - 1]) - chi * y[n] * x[n] + E * x[n]
        I2 = 0.6 * x[n] ** 2 - 0.356 * (-1 + sp.exp(-4.45 * y[n])) * sp.exp(-4.45 * y[n]) + (-0.02 * y[n+1] + 0.02 * y[n]) * sp.exp(
            -0.35 * y[n+1] - 0.35 * y[n]) - 0.0035 * (y[n+1] - y[n]) ** 2 * sp.exp(-0.35 * y[n+1] - 0.35 * y[n]) - 0.0035 * (
                         -y[n-1] + y[n]) ** 2 * sp.exp(-0.35 * y[n-1] - 0.35 * y[n]) + (-0.02 * y[n-1] + 0.02 * y[n]) * sp.exp(
            -0.35 * y[n-1] - 0.35 * y[n])
        I3 = x[n]**2
        E1.append(I1)
        E2.append(I2)
        E3.append(I3)
    H1 = V * (x[98] + x[0]) - chi * y[99] * x[99] + E * x[99]  # * (0+1j))
    H2 = 0.6 * x[99] ** 2 - 0.356 * (-1 + sp.exp(-4.45 * y[99])) * sp.exp(-4.45 * y[99]) + (-0.02 * y[0] + 0.02 * y[99]) * sp.exp(
        -0.35 * y[0] - 0.35 * y[99]) - 0.0035 * (y[0] - y[99]) ** 2 * sp.exp(-0.35 * y[0] - 0.35 * y[99]) - 0.0035 * (
                     -y[98] + y[99]) ** 2 * sp.exp(-0.35 * y[98] - 0.35 * y[99]) + (-0.02 * y[98] + 0.02 * y[99]) * sp.exp(
        -0.35 * y[98] - 0.35 * y[99])
    H3 = x[99]**2
    E1.append(H1)
    E2.append(H2)
    E3.append(H3)
    E4 = sum(E3) - 1
    eq = E1, E2
    eq = np.reshape(eq, (30,))
    eq = np.append([eq], [E4])
    eq = eq.flatten()
    return eq

  
  
  
#sol = scipy.optimize.broyden1(F(s1), s0 , f_tol=1e-14)
#print('s0l = ', sol)

#Jacobian
p1 = F(x,y,E)
p2 = sp.Matrix(p1)
B = p2.jacobian(s1)
values = {'E': E_0, 'x0': x_0[0], 'x1': x_0[1], 'x2': x_0[2], 'x3': x_0[3], 'x4': x_0[4], 'x5': x_0[5], 'x6': x_0[6], 'x7': x_0[7], 'x8': x_0[8], 'x9': x_0[9], 'x10': x_0[10],'x11': x_0[11],  'x12': x_0[12], 'x13': x_0[13], 'x14': x_0[14], 'y0': y_0[0], 'y1': y_0[1], 'y2': y_0[2], 'y3': y_0[3], 'y4': y_0[4], 'y5': y_0[5], 'y6': y_0[6], 'y7': y_0[7], 'y8': y_0[8], 'y9': y_0[9], 'y10': y_0[10],'y11': y_0[11],  'y12': y_0[12], 'y13': y_0[13], 'y14': y_0[14]}
B= B.subs(values)
print('B=', B)
Br  = np.array(B).astype(np.float64)
H = sp.Inverse(B)
print('H=', H)
print('multiplication is =', np.shape(H*F(x_0,y_0,E_0)))
#H = np.array(H).astype(np.float64)
H1 = np.zeros((31, 31, 100), dtype = np.float64)
B1 = np.zeros((31, 31, 100), dtype = np.float64)
t2 = np.zeros((31, 100), dtype = np.float64)
b = np.zeros((31, 100), dtype = np.float64)
l = np.zeros((15, 100), dtype =  np.float64)
m = np.zeros((15, 100), dtype = np.float64)
o = np.zeros((15, 100), dtype = np.float64)
l[:, 0] = s0[0:15]
m[:, 0] = s0[15:31]
o[:, 0] = s0[31]
#B1[:, :, 0] = np.array(H).astype(np.float64)
B1[:, :, 0] = np.array(Br).astype(np.float64)
print('B1=',B1[:,:,0])
for k in range(20):
    print('k=', k)
    print('1=', np.shape( B1[:,:,k]))
    print('2=', np.shape(sp.Matrix(- F(l[:, k], m[:, k], o[:, k]))))
    print('3=', np.shape(- np.array(sp.Matrix(B1[:, :, k]).inv())))
    print('4=', (-sp.Matrix(F(l[:, k], m[:, k], o[:, k])).T)* (sp.Matrix(B1[:, :, k]).inv()))
    print('5=', np.shape(-sp.Matrix(F(l[:, k], m[:, k], o[:, k])).T* (sp.Matrix(B1[:, :, k]).inv())))
    print('6=', np.shape(-sp.Matrix(B1[:, :, k]).inv()* (sp.Matrix(F(l[:, k], m[:, k], o[:, k])))))
    print('7=', -sp.Matrix(B1[:, :, k]).inv()* (sp.Matrix(F(l[:, k], m[:, k], o[:, k]))))
    print('b=' , b[:,k])
    H1[:,:, k] =  np.array(sp.Matrix(B1[:, :, k]).inv())
    #print('H1=',H1[:,:,k])
    #print('H2=',np.transpose(H1[:,:,k]))
    #print('H2=',np.shape(np.transpose(H1[:,:,k])))
    #print('H3=',np.shape(sp.Matrix(B1[:, :, k]).inv()))
    #print('H4=',np.shape(H1[:,:,k]))
    t2[:, k+1]  = t2[:, k] + b[:, k]
    l[:, k+1] = t2[0:15, k+1]
    m[:, k+1] = t2[15:30, k+1]
    o[:, k+1] = t2[31, k+1]
    B1[:, :, k+1] = B1[:, :, k] + (1/(np.transpose(b[:, k])*b[:, k]))*((F(l[:, k+1], m[:, k+1], o[:, k+1]) - F(l[:, k+1], m[:, k+1], o[:, k+1])) - B1[:, :, k]*b[:, k]) * np.transpose(b[:, k])
    print('psi is =', l[:, k+1])'''
