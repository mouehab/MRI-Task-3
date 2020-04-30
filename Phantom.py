import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fftshift, ifftshift, fftn, ifftn, fft2
import math
import sys

# BUTTON_BROWSE
 

# ROTATE RECOVERY

def RF_rotate(theta,phantom,row,col):
    for i in range(row):
        for j in range(col):
            phantom[i,j,:]=rotate(theta,phantom[i,j,:])
    return phantom 

def rotate(theta,phantom):
    RF=([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]) 
    phantom=np.dot(RF,phantom) 
    return phantom

    
def recovery(phantom,t,T1,T2,larmarFreq):
    E1 = math.exp(-t/T1)
    E2 = math.exp(-t/T2)
    A= np.array([[E2,0,0],[0,E2,0],[0,0,E1]])
    B= np.array([0, 0, 1-E1])
    phi =2*math.pi*larmarFreq*t/1000
    Rz=np.array([[math.cos(phi),-math.sin(phi),0],[math.sin(phi),math.cos(phi),0],[0, 0, 1]])
    Afb = np.matmul(A,Rz)
    phantom =np.matmul(Afb,phantom)+B 
    return phantom       

def rotate_recovery(phantom,theta,t,T1,T2,row,col):
    larmarFreqOil= mag_field*42.6
    larmarFreqWater= mag_field*5.8
    k=0
    for i in range(row): 
        for j in range(col):
            if(i<2 and j<2):
                phantom[i,j,:]=recovery(phantom[i,j,:],t,T1[i,j],T2[i,j],larmarFreqOil[k])
            else:
                phantom[i,j,:]=recovery(phantom[i,j,:],t,T1[i,j],T2[i,j],larmarFreqWater[k])
            if k>=999:
                k = 0
            else:
                k=k+1    

    return phantom         

def rotate_decay(phantom,theta,TE,T2,row,col):
    for i in range(row): 
        for j in range(col):
            phantom[i,j,:]=rotate(theta,phantom[i,j,:]) 
            phantom[i,j,:]=decay(phantom[i,j,:],TE,T2[i,j]) 
    return phantom         

def recovery1(phantom,row,col,TR,T1):
    for ph_rowtr in range(row): 
        for ph_coltr in range(col):
            phantom[ph_rowtr,ph_coltr,0]=0
            phantom[ph_rowtr,ph_coltr,1]=0
            phantom[ph_rowtr,ph_coltr,2]=((phantom[ph_rowtr,ph_coltr,2])*np.exp(-TR/T1[ph_rowtr,ph_coltr]))+(1-np.exp(-TR/T1[ph_rowtr,ph_coltr]))
    return phantom
def decay(phantom,TE,T2):
    dec=np.exp(-TE/T2)
    phantom=np.dot(dec,phantom)
    return phantom       

# CREATE PHANTOM
def createPhantom(row,col):
    phantom=np.zeros((row,col,3))
    for i in range(row):
        for j in range(col):
            phantom[i,j,2]=1
            
    return phantom


def spin_Echo():
    flip_angle=90
    theta=np.radians(flip_angle) 
    Gy=2*math.pi
    phantom = createPhantom(row,col)
    Kspace=np.zeros((image.shape[0],image.shape[1]),dtype=np.complex_) 
    t=0
    Check=True
    doubleCheck=False
    phantom=rotate_decay(phantom,np.radians(90),TE/2,T2,row,col)
    while (Check):
        phantom=rotate_recovery(phantom,theta,t,T1,T2,row,col)
        print(phantom)
        t=t+1
        for i in range(row):
            for j in range(col):
                if(phantom[i,j,2]<0.9):
                    Check=True
                    doubleCheck=True
                    break
                else:
                    Check=False
            if (doubleCheck==True):
                break


    phantom=rotate_decay(phantom,np.radians(180),TE/2,T2,row,col)

    for r in range(Kspace.shape[0]):  #rows
        for c in range(Kspace.shape[1]): #columns
            Gx_step=((2*math.pi)/row)*r
            Gy_step=((Gy)/col)*c
            for ph_row in range(row): 
                for ph_col in range(col):
                    Toltal_theta=(Gx_step*ph_row)+(Gy_step*ph_col)
                    Mag=math.sqrt(((phantom[ph_row,ph_col,0])*(phantom[ph_row,ph_col,0]))+((phantom[ph_row,ph_col,1])*(phantom[ph_row,ph_col,1])))
                    Kspace[r,c]=Kspace[r,c]+(Mag*np.exp(-1j*Toltal_theta))


   
    inverse_array = np.fft.fftshift(Kspace)
    originalImage=np.fft.ifftshift(inverse_array)
    originalImage = np.fft.ifft2(originalImage)
    originalImage= np.abs(originalImage)
    originalImage = (originalImage - np.amin(originalImage)) * 255/ (np.amax(originalImage) - np.amin(originalImage))
    inverse_array = np.abs(inverse_array)
    inverse_array = (inverse_array - np.amin(inverse_array)) * 255/ (np.amax(inverse_array) - np.amin(inverse_array))
    cv2.imwrite("original.jpg",originalImage)
    cv2.imwrite("result.jpg",inverse_array)

    


image = cv2.imread(sys.argv[1], 0)

size_image=(len(image))
T1=np.zeros((size_image,size_image))
T2=np.zeros((size_image,size_image))
row=size_image
col=size_image
TR=90
TE= 30
mag_field = np.random.uniform(low= 0 , high= 5, size = 1000)

for i in range(row):
    for j in range(col):

        if (image[i,j] >= 0 and image[i,j] <20):
            T1[i,j]=500
            T2[i,j]=100    
        elif ( image[i,j]>=20 and image[i,j]<180):
            T1[i,j]=1000
            T2[i,j]=120
        elif ( image[i,j]>=180 and image[i,j]<255):
            T1[i,j]=1500
            T2[i,j]=150  
spin_Echo()



# DRAWING
img1=cv2.imread("result.jpg",0)
img2=cv2.imread("original.jpg",0)
print("---------------------------------------")
fig=plt.figure(figsize=(32, 32))
fig.add_subplot(2, 2, 1)
plt.imshow(image, cmap = 'gray', interpolation='nearest')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
fig.add_subplot(2, 2, 2)
plt.imshow(img1, cmap = 'gray' ,interpolation='nearest')
plt.title('K-Space'), plt.xticks([]), plt.yticks([])
fig.add_subplot(2, 2, 3)
plt.imshow(img2, cmap = 'gray' ,interpolation='nearest')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.show()
