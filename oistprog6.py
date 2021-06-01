# creating a dataset for soma detection
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import measure
from skimage.feature import blob_dog, blob_log, blob_doh
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from scipy import *
import os
import glob
import csv
import math
from datetime import datetime
start_time = datetime.now()
# do your work here

#read image from directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".tif"):
            img = cv2.imread(os.path.join(folder, filename))
            plt.savefig('filename.png')
            images.append(img)
    return images
root_folder = ''
folders = [os.path.join(root_folder, x) for x in ('oist1', 'oist2')]
imgs = [img for folder in folders for img in load_images_from_folder(folder)]

#convert images to grayscale images
def load_images(gray_folder):
    gray_images = []
    for filename in os.listdir(gray_folder):
        if filename.endswith(".tif"):
            img = cv2.imread(os.path.join(gray_folder, filename))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #gray_img = cv2.GaussianBlur(img, (5,5), 0)
            gray_images.append(gray_img)
    return gray_images
gray_folders = [os.path.join(root_folder, x) for x in ('oist1', 'oist2')]
gray_imgs = [gray_img for gray_folder in gray_folders for gray_img in load_images(gray_folder)]
for j in range(len(gray_imgs)):
    cv2.imwrite(f'gray/gray_{j}.tif', gray_imgs[j])
    
# apply SLIC and extract (approximately) the supplied number of segments
def sp_idx(s, index=True):
    u = np.unique(s)
    return [np.where(s == i) for i in u]
def numberofsegments():
    b = np.empty((0, 100))
    for j in range(len(imgs)):
        segments_slic = slic(gray_imgs[j], n_segments=500, compactness=10, sigma=1, start_label=1)
        a = len(np.unique(segments_slic))
        b = np.append([b], [a])
    return b
a = numberofsegments()
im_segments_slic = []; vdf = []
im_superpixel = np.empty((0, int(a[j])))
sp_gray_avg = np.empty((0, int(a[j])))
sp_width = np.empty((0, int(a[j]))); sp_height = np.empty((0, int(a[j])))
area = np.empty((0, int(a[j]))); eccentricity = np.empty((0, int(a[j])))
sp_centx = np.empty((0, int(a[j])));sp_centy = np.empty((0, int(a[j]))) #;centroid = np.empty((0, int(a[j])))
im_x = np.empty((0, int(a[j]))); im_y = np.empty((0, int(a[j])))
sp_id = np.empty((0, int(a[j]))); im = np.empty((0, int(a[j])))
im_IMF = np.empty((0, int(a[j]))); im_p1 = np.empty((0, int(a[j]))); im_p2 = np.empty((0, int(a[j]))); im_p3 = np.empty((0, int(a[j])))
im_area = np.empty ((0, int(a[j]))); im_eccentricity = np.empty ((0, int(a[j]))); im_gray_avg = np.empty ((0, int(a[j])))
for j in range(len(gray_imgs)):
    segments_slic = slic(gray_imgs[j], n_segments=500, compactness=10, sigma=1, start_label=1)
    segments_ids = np.unique(segments_slic)
    print('slic segments are', segments_slic)
    print('j=', j)
    #print(f"SLIC number of segments:{len(np.unique(segments_slic))}")
    superpixel_list = sp_idx(segments_slic)
    superpixel = [idx for idx in superpixel_list]
    #print('superpixel[0]=', superpixel[0])
    #print('superpixel[8]=', superpixel[8][0])
    # 932 rows x 932 columns, 932 se unique 484 SLIC segments bana diye... jo same h uska same number rakha #tabhi  1 1 1 1 ....484 484 484 end number bana
    # in a single image 484 unique superpixels h... ab ek superpixel mn 1764  se 2065 pixels vary kar sakte h
    x=[0 for i in range(len(superpixel))]
    y=[0 for i in range(len(superpixel))] 
    #centers = np.array([np.mean(np.nonzero(segments_slic == i), axis=1) for i in segments_ids])
    w = []; h = []; centx = []; centy = []; sp_x = []; sp_y = []; im_no = []
    im_sp_area = []; im_sp_intensity = []; im_sp_eccentricity = []; im_sp_gray_avg = []
    regions = measure.regionprops(segments_slic, intensity_image=gray_imgs[j])
    for r in regions:
        sp_area = r.area
        sp_eccentricity = r.eccentricity
        sp_mean_intensity = r.mean_intensity
        im_sp_area.append(sp_area)
        im_sp_eccentricity.append(sp_eccentricity)
        im_sp_gray_avg.append(sp_mean_intensity)
    rows = []; cols = []
    for segVal in np.unique(segments_slic):  #segval is im_sp_centroid=[] 1,2,3,4,5,6,7,8,9 i.e. superpixels
        #print('segmentslic=',len(np.unique(segments_slic)))
        #print('segval=',segVal)
        mask = np.ones(gray_imgs[j].shape[:2], dtype='uint8') #   self.height, self.width = img.shape[:2]
        mask[segments_slic == segVal] = 255
        pos = np.where(mask == 0)
        x = pos[:][0]  #  XY = np.array([superpixel[i][0], superpixel[i][1]]).T
        y = pos[:][1]
        ymin = np.min(pos[:][1]);ymax = np.max(pos[:][1])
        xmin = np.min(pos[:][0]);xmax = np.max(pos[:][0])
        cx = np.mean(x); cy = np.mean(y)
        width = xmax - xmin + 1;w.append(width)
        height = ymax - ymin + 1;h.append(height)
        sp_x.append(x);sp_y.append(y)
        centx.append(cx);centy.append(cy)
        im_no.append(j)
        #centerx,centery = [np.mean(np.nonzero(segments_slic == i), axis=1) for i in segments_ids]
 # making a  sliding window
        row = np.size(superpixel[segVal-1][0])
        col =  np.size(superpixel[segVal-1][1])
        rows.append(row);cols.append(col)
        #print('rows=',rows);print('cols=',cols)
        f1 = [];f3 = [];f2 = []
        #print('shape of superpixel=', np.shape(superpixel[segVal-1]))
    sx  = pd.DataFrame(sp_x).to_csv('./xData.csv')
    sy  = pd.DataFrame(sp_y).to_csv('./yData.csv')
    bx = pd.read_csv('./xData.csv')
    #cx  = pd.DataFrame(bx.loc[6]).to_csv('./6xdata.csv')
    by = pd.read_csv('./yData.csv')
    #print('3rd value = ', bx.iloc[6]);print('6th row=', rows[6]);print('sp_x=', sp_x[5]);print('sp_y=', sp_y[5])
    sp_mask2 = []; sp_cent_px = []
    f3 =[]; f4 = []; p = []; q = []
    for segVal in np.unique(segments_slic):
        print('seVal=', segVal) 
        i = 0; f1 = []
        for a1, b1, c1, d1, e1, a2, b2, c2, d2, e2 in zip(bx.iloc[segVal-1 ,1:rows[segVal-1]].astype(int), bx.iloc[segVal-1, 2:rows[segVal-1]-1].astype(int), bx.iloc[segVal-1 ,3:rows[segVal-1]-2].astype(int), bx.iloc[segVal-1, 4:rows[segVal-1]-3].astype(int), bx.iloc[segVal-1 ,5:rows[segVal-1]-4].astype(int), by.iloc[segVal-1, 1:rows[segVal-1]].astype(int), by.iloc[segVal-1 ,2:rows[segVal-1]-1].astype(int), by.iloc[segVal-1, 3:rows[segVal-1]-2].astype(int), by.iloc[segVal-1 ,4:rows[segVal-1]-3].astype(int), by.iloc[segVal-1, 5:rows[segVal-1]-4].astype(int)):
            '''print('m+2=', e1);print('n+2=',e2);print('m+1=', d1);print('n+1=', d2)
            print('m=', c1);print('n=', c2);print('m-1=', b1);print('n-1=', b2)
            print('m-2=', a1);print('n-2=', a2)'''
            mask2 = np.array([gray_imgs[j][a1, a2], gray_imgs[j][b1, a2], gray_imgs[j][b1, b2], gray_imgs[j][c1, a2], gray_imgs[j][c1, b2], gray_imgs[j][c1, c2], gray_imgs[j][c1, d2], gray_imgs[j][d1, b2], gray_imgs[j][d1, c2], gray_imgs[j][d1, d2], gray_imgs[j][d1, e2], gray_imgs[j][e2, d2], gray_imgs[j][e1, e2]])
            cent_px = [gray_imgs[j][c1, c2]]
            #print('mask2=', mask2); print('center_pixel=', cent_px)
            f = []
            for k in range(len(mask2)):
                #print('k=',k) 
                p_i = cent_px; pj = mask2; p_j = pj[k]
                if abs(p_i-p_j) <= 3:
                    e1 = 0
                    f.append(e1)
                if 3<abs(p_i-p_j) <= 12:
                    e2 = math.exp(abs(p_i-p_j)/3)
                    f.append(e2)
                if abs(p_i-p_j) > 12:
                    e3 = math.exp(4)
                    f.append(e3)
                    #print('f=',f)
            i += 1
            p.append(mask2); q.append(cent_px)
            f1.append(f)
        print('i=', i)
        #print('f1=', f1)
        print('f1 final=', np.shape(f1))
        IMF = np.sum(f1)/(13*len(superpixel[segVal-1][0]))
        print('IMF=', IMF)
        print('00000000000000000000000')
        '''p1=[];p2=[];p3=[]
        if IMF > 40:
            P_ST = superpixel[segVal-1]
            p1.append(P_ST)
        elif IMF < 20:
            P_UD = superpixel[segVal-1]
            p2.append(P_UD)
        else:
            P_OB = superpixel[segVal-1]
            p3.append(P_OB)
        print('p3=',p3)
        f4.append(IMF)
        print('f4=', f4)
        #f5.append(IMF); print('f5=',f5)
        p_up = np.max(p3); p_fl = np.min(p3)
        d_p = (p_up-p_fl)/len(p3)
# if we devide ith superpixel in P_OB into 10 parts. and calculate ...
#number of pixel in each part and then find one with max number of pixel
# and the second max and then define function A and theta and thus calculate u1 and u2
    for j in range(1,len(p3)):
        sp_segments_slic = slic(p3[j], n_segments=10, compactness=10, sigma=1, start_label=1)
        print('*****************')
        sp_pixel_list = sp_idx(sp_segments_slic)
        sp_pixel = [idx for idx in sp_pixel_list]
        print('so_pixel_list[',j,']=',sp_pixel_list)
        print('sp_pixel[','j',']=',sp_pixel)
        l = []
        for k in range(np.unique(sp_segments_slic)):
            l1 = len(sp_pixel)
            l.append(l)
            sp_pixmax = np.max(l)
            print('number of maximum pixel are=', sp_pixmax)
            print('\n')
            sp_2_pixmax = np.max(l-sp_pixmax)
            print('second number of maximum pixel are=', sp_2_pixmax) '''
    #im_p1 = np.append([im_p1],[p1]); im_p2 = np.append([im_p2],[p2]); im_p3 = np.append([im_p3],[p3])
    im_IMF = np.append([im_IMF], [f4])
    im = np.append([im], [im_no]); sp_id = np.append([sp_id], [segments_ids])
    im_x = np.append([im_x], [sp_x]);im_y = np.append([im_y], [sp_y])
    sp_centx = np.append([sp_centx], [centx]);sp_centy = np.append([sp_centy], [centy])
    sp_width = np.append([sp_width], [w]);sp_height = np.append([sp_height], [h])
    im_segments_slic.append(segments_slic); im_gray_avg = np.append([im_gray_avg],[im_sp_gray_avg])
    im_area = np.append([im_area],[im_sp_area]); im_eccentricity = np.append([im_eccentricity],[im_sp_eccentricity])
z1 = pd.DataFrame(im_area).to_csv('./im_area.csv')
z2 = pd.DataFrame(im_eccentricity).to_csv('./im_eccentricity.csv')
z3 = pd.DataFrame(im_IMF).to_csv('./im_IMF.csv')
#z4 = np.column_stack([im_p1, im_p2, im_p3])
#zdf = pd.Dataframe(z2, columns=['p1', 'p2', 'p3']).to_csv('./pixelclass' , sep=',', index=false, header=True)
z = pd.DataFrame(im_gray_avg).to_csv('./im_gray_avg.csv')
v = np.column_stack([im, sp_id, sp_centx, sp_centy, sp_width, sp_height, im_area, im_gray_avg, im_eccentricity])
df = pd.DataFrame(v, columns=['Img no', 'sp_id', 'cent_X', 'cent_Y', 'width', 'height', 'area', 'grayavg', 'eccentricity'])
vdf.append(df)
fdf = pd.concat(vdf).to_csv('./im_sp_data.csv', sep=',', index=False, header=True)
'''
#os.chdir("imgdir")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
for j in range(len(gray_imgs)):
    for filename in '/flash/TerenzioU/imgdir':
        d = pd.read_csv(r'/flash/TerenzioU/imgdir/gray_{}.tif.csv'.format(j))
        d['Img no'] = j
#        d.to_csv(r'/home/s/shubhangi-goyal/imgdir/gray_{j}.tif.csv', index=False)
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv('./flash/TerennzioU/img_data.csv', index=False, encoding='utf-8-sig')
#df1 = pd.read_csv('/home/s/shubhangi-goyal/sp_data.csv')
#df1 = df1.drop(['height', 'width', 'eccentricity', 'sp_id'], axis=1)
#df2 = pd.read_csv('./img_data.csv')
#df2 = df2.drop(['Type', 'Unnamed: 8', 'R', 'Confidence', 'Contrast', 'SNR', 'Type', 'img number'], axis=1)
#L1 = df1[['X', 'Y']].apply(tuple, axis=1).tolist()
#L2 = df2[['X', 'Y']].apply(tuple, axis=1).tolist()
#soma = [i for i in L1 if i in L2]'''
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
