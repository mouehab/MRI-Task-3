import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fftshift, ifftshift, fftn, ifftn, fft2
from qimage2ndarray import gray2qimage  
import math
import sys
import os

TR= 0.09
TE= 0.03
Gy= 2*math.pi
# BUTTON_BROWSE
imagepath= sys.argv[1]
image = cv2.imread(imagepath, 0)
size_image=(len(image))

T1=np.zeros((size_image,size_image))
T2=np.zeros((size_image,size_image))
for i in range(size_image):
    for j in range(size_image):

        if (image[i,j] >= 0 and image[i,j] <20):
            T1[i,j]=500
            T2[i,j]=100    
        elif ( image[i,j]>=20 and image[i,j]<180):
            T1[i,j]=1000
            T2[i,j]=120
        elif ( image[i,j]>=180 and image[i,j]<255):
            T1[i,j]=1500
            T2[i,j]=150  

# print(T2) 


# ROTATE DECAY
def RF_rotate(theta,phantom,row,col):
    for i in range(row):
        for j in range(col):
            phantom[i,j,:]=rotate(theta,phantom[i,j,:])
            
    return phantom 

def rotate(theta,phantom):
    RF=([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]) 
    phantom=np.dot(RF,phantom) 
    return phantom

    
def decay(phantom,TE,T2):
    # print(T2)
    dec=np.exp(-TE/T2)
    phantom=np.dot(dec,phantom)
    return phantom       

def rotate_decay(phantom,theta,TE,T2,row,col):
    print("rotate_decay")
    for i in range(row): 
        for j in range(col):
            phantom[i,j,:]=rotate(theta,phantom[i,j,:]) 
            phantom[i,j,:]=decay(phantom[i,j,:],TE,T2[i,j]) 
    return phantom         

# CREATE PHANTOM
def createPhantom(row,col):
    phantom=np.zeros((row,col,3))
    for i in range(row):
        for j in range(col):
            # phantom[i,j,2]=np.random.randint(low=0, high=1, size=1)
            # phantom[i,j,1]=np.random.randint(low=0, high=1, size=1)
            # phantom[i,j,0]=np.random.randint(low=0, high=1, size=1)
            
            phantom[i,j,2]= 1
           

    return phantom

def recovery(phantom,row,col,TR,T1):
    print("recovery")
    for ph_rowtr in range(row): 
        for ph_coltr in range(col):
            phantom[ph_rowtr,ph_coltr,0]=0
            phantom[ph_rowtr,ph_coltr,1]=0
            phantom[ph_rowtr,ph_coltr,2]=((phantom[ph_rowtr,ph_coltr,2])*np.exp(-TR/T1[ph_rowtr,ph_coltr]))+(1-np.exp(-TR/T1[ph_rowtr,ph_coltr]))
    return phantom 



def spin_Echo():
    row=size_image
    col=size_image
    theta=np.radians(90) 
    phantom = createPhantom(row,col)
    print(phantom)
    # TE=int(self.ui.TE_Edit.text())
    # TR=int(self.ui.TR_Edit.text())
    Kspace_SE=np.zeros((image.shape[0],image.shape[1]),dtype=np.complex_)
    
    for r in range(Kspace_SE.shape[0]):  #rows
        # print("marwa")
        phantom=rotate_decay(phantom,np.radians(90),TE/2,T2,row,col)
        phantom=recovery(phantom,row,col,TE/2,T1)
        phantom=rotate_decay(phantom,np.radians(180),TE/2,T2,row,col)
        for c in range(Kspace_SE.shape[1]):
            # print("ehab")
            Gx_step=((2*math.pi)/row)*r
            Gy_step=(Gy/col)*c
            for ph_row in range(row): 
                for ph_col in range(col):
                    # print("allaa")
                    Toltal_theta=(Gx_step*ph_row)+(Gy_step*ph_col)
                    Mag=math.sqrt(((phantom[ph_row,ph_col,0])*(phantom[ph_row,ph_col,0]))+((phantom[ph_row,ph_col,1])*(phantom[ph_row,ph_col,1])))
                    
                    Kspace_SE[r,c]=Kspace_SE[r,c]+(Mag*np.exp(-1j*Toltal_theta))
                    # QApplication.processEvents()
                    
            # QApplication.processEvents()
            
        phantom=recovery(phantom,row,col,TR,T1)  
        # print(phantom)
        
        # QApplication.processEvents()
    
    Kspace= np.fft.fftshift(Kspace_SE)
    Kspace = np.abs(Kspace)
    Kspace = (Kspace - np.amin(Kspace)) * 255/ (np.amax(Kspace) - np.amin(Kspace))
    cv2.imwrite("Kspace.jpg",Kspace)

    inverse_array = np.fft.ifftshift(Kspace_SE)
    inverse_array = np.fft.ifft2(inverse_array)
    inverse_array= np.abs(inverse_array)
    inverse_array = (inverse_array - np.amin(inverse_array)) * 255/ (np.amax(inverse_array) - np.amin(inverse_array))
    cv2.imwrite("Inverse.jpg",inverse_array)

spin_Echo()


# DRAWING
img=cv2.imread("Kspace.jpg",0)
# print(img)
print("---------------------------------------")
# print(image)
plt.subplot(121),plt.imshow(image, cmap = 'gray', interpolation='nearest')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img, cmap = 'gray' ,interpolation='nearest')
plt.title('K-Space'), plt.xticks([]), plt.yticks([])
plt.show()

# DRAWING
img2=cv2.imread("Inverse.jpg",0)
# print(img)
print("---------------------------------------")
# print(image)
plt.subplot(121),plt.imshow(image, cmap = 'gray', interpolation='nearest')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img2, cmap = 'gray' ,interpolation='nearest')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.show()












# img = cv2.imread('brain.jpg', 0)
# dim = range(img.ndim)

# k = fftshift(fftn(ifftshift(img, axes=dim), s=None, axes=dim), axes=dim)
# k /= np.sqrt(np.prod(np.take(img.shape, dim)))
# k = np.real(k)    
# magnitude_spectrum = 20*np.log(np.abs(k)+1)
