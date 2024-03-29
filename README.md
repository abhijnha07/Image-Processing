# Image-Processing

# 1.Python Program to display GrayScale image using read() and write() operations

pip install opencv-python</br>
import cv2</br>
img=cv2.imread('flower3.jpg',0)</br>
cv2.imshow('Window',img)</br>
cv2.waitKey(0)</br>
cv2.destroyAllWindows()</br>

![image](https://user-images.githubusercontent.com/97939934/173807678-011c312b-e4d6-409f-9366-b4a53721422e.png)



# 2.Program to perform linear transformation

from PIL import Image</br>
img=Image.open("bfly3.webp")</br>
img=img.rotate(180)</br>
img.show()</br>
cv2.waitKey(0)</br>
cv2.destroyAllWindows()</br>

![image](https://user-images.githubusercontent.com/97939934/175022026-b46db927-a3c0-46b8-bb85-53dee2797ba5.png)



# 3.Program to display image using matplotlib

import matplotlib.image as mping</br>
import matplotlib.pyplot as plt</br>
img=mping.imread('flower3.jpg')</br>
plt.imshow(img)</br>

![image](https://user-images.githubusercontent.com/97939934/173808767-1e2dc6a1-a6de-427a-9bfb-d3bf33344cfa.png)</br>



# 4.Program to convert color string to RGB color values

from PIL import ImageColor</br>
img1=ImageColor.getrgb("yellow")</br>
print(img1)</br>
img2=ImageColor.getrgb("red")</br>
print(img2)</br>

OUTPUT:</br>
(255, 255, 0)</br>
(255, 0, 0)</br>



# 5.Program to create image using colors

from PIL import Image</br>
img=Image.new('RGB',(300,400),(255,0,255))</br>
img.show()</br>

![image](https://user-images.githubusercontent.com/97939934/173809661-33f675b1-d478-48fd-b0c4-81780098c45d.png)</br>



# 6.Program to visualize images using various color spaces

import cv2</br>
import matplotlib.pyplot as plt</br>
import numpy as np</br>
img=cv2.imread('flower2.webp')</br>
plt.imshow(img)</br>
plt.show()</br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)</br>
plt.imshow(img)</br>
plt.show()</br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)</br>
plt.imshow(img)</br>
plt.show()</br>

![image](https://user-images.githubusercontent.com/97939934/173810745-d3539ea3-94cb-40cc-9813-6e40e31467a1.png)
![image](https://user-images.githubusercontent.com/97939934/173813167-9424d45b-bb31-4abf-983d-9e17308a5e58.png)
![image](https://user-images.githubusercontent.com/97939934/173813279-d5643224-b653-4fe6-a897-4139e8615e88.png)</br>



# 7.Program to display image attributes

from PIL import Image</br>
image=Image.open('flower2.webp')</br>
print("FileName:",image.filename)</br>
print("Format:",image.format)</br>
print("Mode:",image.mode)</br>
print("Size:",image.size)</br>
print("Width:",image.width)</br>
print("Height:",image.height)</br>

OUTPUT:</br>
FileName: flower2.webp</br>
Format: WEBP</br>
Mode: RGB</br>
Size: (263, 300)</br>
Width: 263</br>
Height: 300</br>



# 8.Program to convert the original image to gray scale and then to binary

import cv2</br>
#read the image file</br>
img=cv2.imread('leaf1.jpg')</br>
cv2.imshow("RGB",img)</br>
cv2.waitKey(0)</br>
#Grayscale</br>
img=cv2.imread('leaf1.jpg',0)</br>
cv2.imshow("Gray",img)</br>
cv2.waitKey(0)</br>
#Binary Image</br>
ret, bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)</br>
cv2.imshow('Binary', bw_img)</br>
cv2.waitKey(0)</br>
cv2.destroyAllWindows()</br>

![image](https://user-images.githubusercontent.com/97939934/174047484-01102d3e-4909-4f0c-83c0-424792eccdfe.png)
![image](https://user-images.githubusercontent.com/97939934/174047586-f1a7b9d7-9f6a-468c-a3b0-8530b47d28de.png)
![image](https://user-images.githubusercontent.com/97939934/174047676-4c0667bb-7ec8-4773-97f8-243e48496624.png)</br>



# 9.Program to resize the original image</br>

import cv2</br>
img=cv2.imread('flower2.jpg')</br>
print('Original image length width',img.shape)</br>
cv2.imshow('Original image',img)</br>
cv2.waitKey(0)</br>
#to show the resized image</br>
imgresize=cv2.resize(img,(150,160))</br>
cv2.imshow('Resized Image',imgresize)</br>
print('Resized image length width',imgresize.shape)</br>
cv2.waitKey(0)</br>

OUTPUT:</br>
Original image length width (531, 800, 3)</br>
Resized image length width (160, 150, 3)</br>

![image](https://user-images.githubusercontent.com/97939934/174047929-d54ef7e2-89c5-4261-9b11-564569b1c038.png)</br>
![image](https://user-images.githubusercontent.com/97939934/174048162-bd46ded8-67eb-4266-a22d-32c0df78bdf1.png)</br>



# 10.Develop a program to readimage using URL.</br>

from skimage import io</br>
import matplotlib.pyplot as plt</br>
url='https://climate.nasa.gov/rails/active_storage/blobs/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBbjRyIiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--0a7f9ec62ad04559ccea084556300e01789e456a/9827327865_98e0f0dc2d_o.jpg'</br>
image=io.imread(url)</br>
plt.imshow(image)</br>
plt.show()</br>
![image](https://user-images.githubusercontent.com/97939934/175019786-b1b4feb0-2990-49ab-9804-f3b3bc86999d.png)</br>



# 11.Write a program to mask and blur the image</br>

import cv2</br>
import matplotlib.image as mpimg</br>
import matplotlib.pyplot as plt</br>
img=mpimg.imread("leaf1.jpg")</br>
plt.imshow(img)</br>
plt.show()</br>

![image](https://user-images.githubusercontent.com/97939934/175020010-42f3f37e-5582-4350-89e4-d7f0935ddf7a.png)</br>

hsv_img=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)</br>
light_orange=(1,190,200)</br>
dark_orange=(18,255,255)</br>
mask=cv2.inRange(hsv_img,light_orange,dark_orange)</br>
result=cv2.bitwise_and(img,img,mask=mask)</br>
plt.subplot(1,2,1)</br>
plt.imshow(mask,cmap='gray')</br>
plt.subplot(1,2,2)</br>
plt.imshow(result)</br>
plt.show()</br>

![image](https://user-images.githubusercontent.com/97939934/175020098-0507bfa7-589e-4494-bee6-47c6dd3e87d6.png)</br>

light_white=(0,0,200)</br>
dark_white=(145,60,255)</br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)</br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)</br>
plt.subplot(1,2,1)</br>
plt.imshow(mask_white,cmap='gray')</br>
plt.subplot(1,2,2)</br>
plt.imshow(result_white)</br>
plt.show()</br>

![image](https://user-images.githubusercontent.com/97939934/175020184-1d5e5429-767f-4e79-8a4e-0d00f31d46b1.png)</br>

final_mask=mask+mask_white</br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)</br>
plt.subplot(1,2,1)</br>
plt.imshow(final_mask,cmap='gray')</br>
plt.subplot(1,2,2)</br>
plt.imshow(final_result)</br>
plt.show()</br>

![image](https://user-images.githubusercontent.com/97939934/175020299-7654888a-68d1-4b9c-9413-77864927fdb5.png)</br>

blur=cv2.GaussianBlur(final_result,(7,7),0)</br>
plt.imshow(blur)</br>
plt.show()</br>

![image](https://user-images.githubusercontent.com/97939934/175020508-92a308c8-b349-405b-9c69-318fe2553076.png)</br>



# 12.Write a program to perform arithmatic operations on images.</br>

import cv2</br>
import matplotlib.image as mpimg</br>
import matplotlib.pyplot as plt</br>
img1=cv2.imread('flower3.jpg')</br>
img2=cv2.imread('leaf1.jpg')</br>
fimg1=img1+img2</br>
plt.imshow(fimg1)</br>
plt.show()</br>
cv2.imwrite('output.jpg',fimg1)</br>
fimg2=img1-img2</br>
plt.imshow(fimg2)</br>
plt.show()</br>
cv2.imwrite('output.jpg',fimg2)</br>
fimg3=img1*img2</br>
plt.imshow(fimg3)</br>
plt.show()</br>
cv2.imwrite('output.jpg',fimg3)</br>
fimg4=img1/img2</br>
plt.imshow(fimg4)</br>
plt.show()</br>
cv2.imwrite('output.jpg',fimg4)</br>

![image](https://user-images.githubusercontent.com/97939934/175020761-bc0131d3-1b12-4a4f-871e-46df65ad9e75.png)
![image](https://user-images.githubusercontent.com/97939934/175020852-952dd631-c406-4f81-bb58-0258d6a599a6.png)
![image](https://user-images.githubusercontent.com/97939934/175020936-9674ffc1-8f29-4fa2-be6f-586ce54fa1c3.png)</br>



# 13. Develop the program to change the image to different color spaces</br>
 
import cv2</br>
img=cv2.imread('D:\\flower0.jpg')</br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)</br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)</br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)</br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)</br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)</br>
cv2.imshow("GRAY image",gray)</br>
cv2.imshow("HSV image",hsv)</br>
cv2.imshow("LAB image",lab)</br>
cv2.imshow("HLS image",hls)</br>
cv2.imshow("YUV image",yuv)</br>
cv2.waitKey(0)</br>
cv2.destroyAllWindows()</br>

![image](https://user-images.githubusercontent.com/97939934/175267640-9245cbcc-f2f1-4d78-b291-9b5b1fed46f1.png)</br>
![image](https://user-images.githubusercontent.com/97939934/175267823-391abf88-72b0-4276-adad-6004c567be65.png)</br>



# 14. Program to create an image using 2D array</br>

import cv2 as c</br>
import numpy as np</br>
from PIL import Image</br>
array=np.zeros([100,200,3],dtype=np.uint8)</br>
array[:,:100]=[255,130,0]</br>
array[:,100:]=[0,0,255]</br>
img=Image.fromarray(array)</br>
img.save('Image1.png')</br>
img.show()</br>
c.waitKey(0)</br>

![image](https://user-images.githubusercontent.com/97939934/175280445-f94934d7-ddde-46ef-a773-0cc12c7a2b54.png)</br>



# 15.Program to implement Bitwise operations.</br>

import cv2</br>
import matplotlib.pyplot as plt</br>
image1=cv2.imread('b1.jpg')</br>
image2=cv2.imread('b1.jpg')</br>
ax=plt.subplots(figsize=(15,10))</br>
bitwiseAnd=cv2.bitwise_and(image1,image2)</br>
bitwiseOr=cv2.bitwise_or(image1,image2)</br>
bitwiseXor=cv2.bitwise_xor(image1,image2)</br>
bitwiseNot_img1=cv2.bitwise_not(image1)</br>
bitwiseNot_img2=cv2.bitwise_not(image2)</br>
plt.subplot(151)</br>
plt.imshow(bitwiseAnd)</br>
plt.subplot(152)</br>
plt.imshow(bitwiseOr)</br>
plt.subplot(153)</br>
plt.imshow(bitwiseXor)</br>
plt.subplot(154)</br>
plt.imshow(bitwiseNot_img1)</br>
plt.subplot(155)</br>
plt.imshow(bitwiseNot_img2)</br>
cv2.waitKey(0)</br>

![image](https://user-images.githubusercontent.com/97939934/176425366-d3455c3d-0bc5-497b-8bd5-6d7ba573392f.png)</br>



# 16.Program to implements varoius Blur Techniques</br>

import cv2</br>
import numpy as np</br>
image=cv2.imread('bird1.jpg')</br>
cv2.imshow('Original Image', image)</br>
cv2.waitKey(0)</br>
#Gaussian blur    </br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)</br>
cv2.imshow('Gaussian Blurring', Gaussian)</br>
cv2.waitKey(0)</br>
#Median Blur</br>
median=cv2.medianBlur(image,5)</br>
cv2.imshow('Median Blurring', median)</br>
cv2.waitKey(0)</br>
#Bilateral Blurring</br>
bilateral=cv2.bilateralFilter(image,9,75,75)</br>
cv2.imshow('Bilateral Blurring', bilateral)</br>
cv2.waitKey(0)</br>
cv2.destroyAllWindows()</br>

![image](https://user-images.githubusercontent.com/97939934/176425915-c203002b-ea2c-4493-b857-ad1f432e5b6c.png)
![image](https://user-images.githubusercontent.com/97939934/176426055-804fea45-de0e-4a18-a1f7-a36691f57f80.png)</br>
![image](https://user-images.githubusercontent.com/97939934/176426147-56966a08-d998-4b60-a113-efdcb368424d.png)
![image](https://user-images.githubusercontent.com/97939934/176426237-f19f0d1f-6a66-4c8e-bb24-898541666128.png)</br>



# 17.Program to perform Image Enhancement</br>

from PIL import Image</br>
from PIL import ImageEnhance</br>
image=Image.open('img2.jpg')</br>
image.show()</br>
enh_bri=ImageEnhance.Brightness(image)</br>
brightness=1.5</br>
image_brightened=enh_bri.enhance(brightness)</br>
image_brightened.show()</br>
enh_col=ImageEnhance.Color(image)</br>
color=1.5</br>
image_colored=enh_col.enhance(color)</br>
image_colored.show()</br>
enh_con=ImageEnhance.Contrast(image)</br>
contrast=1.5</br>
image_contarsted=enh_con.enhance(contrast)</br>
image_contarsted.show()</br>
enh_sha=ImageEnhance.Sharpness(image)</br>
sharpness=3.0</br>
image_sharped=enh_sha.enhance(sharpness)</br>
image_sharped.show()</br>

![image](https://user-images.githubusercontent.com/97939934/178463059-625deccc-293d-4b8b-b357-ff41cac5e5cc.png)</br>
![image](https://user-images.githubusercontent.com/97939934/178464708-099fa2dd-e028-45ce-b026-2da241e4ed09.png)</br>



# 18.Program to perfrom Morphological operations</br>

import cv2</br>
import numpy as np</br>
from matplotlib import pyplot as plt</br>
from PIL import Image,ImageEnhance</br>
img=cv2.imread('b1.jpg')</br>
ax=plt.subplots(figsize=(20,10))</br>
kernel=np.ones((5,5),np.uint8)</br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)</br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)</br>
erosion=cv2.erode(img,kernel,iterations=1)</br>
dilation=cv2.dilate(img,kernel,iterations=1)</br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)</br>
plt.subplot(151)</br>
plt.imshow(opening)</br>
plt.subplot(152)</br>
plt.imshow(closing)</br>
plt.subplot(153)</br>
plt.imshow(erosion)</br>
plt.subplot(154)</br>
plt.imshow(dilation)</br>
plt.subplot(155)</br>
plt.imshow(gradient)</br>
cv2.waitKey(0)</br>

![image](https://user-images.githubusercontent.com/97939934/176427695-80a9720d-0c5b-4cd5-a864-838013d3a54d.png)</br>



# 19.Develop a program to i</br>
   i)Read the image</br>
   ii)Write(save) the grayscale image and</br>
   iii)Display the original image and grayscale image</br>
   
import cv2</br>
originalImg=cv2.imread('eagle.jpg')</br>
grayImg=cv2.imread('eagle.jpg',0)</br>
isSaved=cv2.imwrite('D:/i.jpg',grayImg)</br>
cv2.imshow('Display Original Image',originalImg)</br>
cv2.imshow('Display GrayScale Image',grayImg)</br>
cv2.waitKey(0)</br>
cv2.destroyAllWindows()</br>
if isSaved:</br>
    print('The Image Is Successfully saved')</br>
    
    
OUTPUT:</br>
The Image Is Successfully saved</br>

![image](https://user-images.githubusercontent.com/97939934/178706979-8397f75e-447f-4fac-9817-00ca5a3873de.png)</br>
![image](https://user-images.githubusercontent.com/97939934/178707184-5b9c507e-5d28-4485-bc78-27495edd6da0.png)</br>



# 20.Program to perform slicing with background</br>

import cv2</br>
import numpy as np</br>
from matplotlib import pyplot as plt</br>
image=cv2.imread('b1.jpg',0)</br>
x,y=image.shape</br>
z=np.zeros((x,y))</br>
for i in range(0,x):</br>
    for j in range(0,y):</br>
        if(image[i][j]>50 and image[i][j]<150):</br>
            z[i][j]=255</br>
        else:</br>
            z[i][j]=image[i][j]</br>
equ=np.hstack((image,z))</br>
plt.title('Graylevel slicing with background')</br>
plt.imshow(equ,'gray')</br>
plt.show()</br>

![image](https://user-images.githubusercontent.com/97939934/178709427-11a328f7-b48c-44ee-8675-46a91af8af37.png)</br>



# 21.Program to perform slicing without background</br>

import cv2</br>
import numpy as np</br>
from matplotlib import pyplot as plt</br>
image=cv2.imread('b1.jpg',0)</br>
x,y=image.shape</br>
z=np.zeros((x,y))</br>
for i in range(0,x):</br>
    for j in range(0,y):</br>
        if(image[i][j]>50 and image[i][j]<150):</br>
            z[i][j]=255</br>
        else:</br>
            z[i][j]=0</br>
equ=np.hstack((image,z))</br>
plt.title('Graylevel slicing without background')</br>
plt.imshow(equ,'gray')</br>
plt.show()</br>

![image](https://user-images.githubusercontent.com/97939934/178709738-7897c6a5-dd22-498a-923d-d802877de2ec.png)</br>


# 22.Analyze the image data using Histogram</br>

#Using skimage library</br>
from skimage import io</br>
import matplotlib.pyplot as plt</br>
image = io.imread('b1.jpg')</br>
</br>
_ = plt.hist(image.ravel(), bins = 256, color = 'orange', )</br>
_ = plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)</br>
_ = plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)</br>
_ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)</br>
_ = plt.xlabel('Intensity Value')</br>
_ = plt.ylabel('Count')</br>
_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])</br>
plt.show()</br>
![image](https://user-images.githubusercontent.com/97939934/178960207-52281a8d-6827-436c-962c-cc80b52ec8ea.png)</br>
</br>
or
</br>
from skimage import io</br>
import matplotlib.pyplot as plt</br>
image = io.imread('b1.jpg')</br>
ax = plt.hist(image.ravel(), bins = 256)</br>
plt.show()</br>
![image](https://user-images.githubusercontent.com/97939934/178960355-2da2de71-cd49-4edb-b130-df6066c8d39a.png)</br>
</br>
#Using opencv</br>
import cv2</br>
from matplotlib import pyplot as plt</br>
img = cv2.imread('b1.jpg',0) #of grayscale image</br>
plt.hist(img.ravel(),256,[0,256])</br>
plt.show()</br>
![image](https://user-images.githubusercontent.com/97939934/178960647-fbe5ad70-261e-4d0e-b66c-b904e424c82f.png)</br>
</br>
#Using numpy</br>
import cv2</br>
import numpy as np</br>
img=cv2.imread('b1.jpg')</br>
hist=cv2.calcHist([img],[0],None,[256],[0,256])</br>
plt.hist(img.ravel(),256,[0,256])</br>
plt.show()</br>
![image](https://user-images.githubusercontent.com/97939934/178960875-24b7cafb-367e-4e1c-8896-ac3c4f2ecaa8.png)</br>

# 23. Program to perform basic image data analysis using intensity transformation</br>
    a) Image Negative    b) Log transformation   c) Gamma Correction</br>

%matplotlib inline</br>
import imageio</br>
import matplotlib.pyplot as plt</br>
#import warnings</br>
#import matplotlib.cbook</br>
#warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)</br>
pic=imageio.imread('nature.jpg')</br>
plt.figure(figsize=(6,6))</br>
plt.imshow(pic);</br>
plt.axis('off');</br>
![image](https://user-images.githubusercontent.com/97939934/179968654-fc1528c2-b118-4e2e-9aae-7f688cef96a7.png)</br>
</br>
negative=255-pic</br>
plt.figure(figsize=(6,6))</br>
plt.imshow(negative);</br>
plt.axis('off');</br>
![image](https://user-images.githubusercontent.com/97939934/179968716-2953c752-b72b-4c87-8d87-212c923d92c9.png)</br>

</br>
%matplotlib inline</br>
import imageio</br>
import numpy as np</br>
import matplotlib.pyplot as plt</br>

pic=imageio.imread('b1.jpg')</br>
gray=lambda rgb : np.dot(rgb[...,:3],[0.299,0.587,0.114])</br>
gray=gray(pic)</br>
</br>
max_=np.max(gray)</br>
</br>
def log_transform():</br>
    return(255/np.log(1+max_))*np.log(1+gray)</br>
plt.figure(figsize=(5,5))</br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray')) # colors can be changed i place of gray</br>
plt.axis('off');</br>
![image](https://user-images.githubusercontent.com/97939934/179961975-1f39b0c0-ff7f-497c-8030-5c47c4d0cd5b.png)</br>
</br>
import imageio</br>
import matplotlib.pyplot as plt</br>
#Gamma encoding</br>
pic=imageio.imread('b1.jpg')</br>
gamma=0.2 #Gamma < 1 ~ Dark; Gamma > 1 ~ Bright</br>
</br>
gamma_correction=((pic/255)**(1/gamma))</br>
plt.figure(figsize=(5,5))</br>
plt.imshow(gamma_correction)</br>
plt.axis('off');</br>
![image](https://user-images.githubusercontent.com/97939934/179962124-8952836c-f3ed-45a6-b738-7b69369b5f76.png)</br>
</br>
# 24. Program to perform basic image manipulation</br>
  a) Sharpness   b) Flipping  c) Cropping</br>
  </br>
#image sharpen</br>
from PIL import Image</br>
from PIL import ImageFilter</br>
import matplotlib.pyplot as plt</br>
#load image</br>
my_image=Image.open('cat.jpg')</br>
#use sharpen function</br>
sharp=my_image.filter(ImageFilter.SHARPEN)</br>
#Save the image</br>
sharp.save("D:/image_sharpen.jpg") #Image is saved in D drive</br> 
sharp.show()</br>
plt.imshow(sharp)</br>
plt.show()</br>
![image](https://user-images.githubusercontent.com/97939934/179968252-ece8e5a4-1153-4a0c-bf69-14cfe1d86500.png)</br>

#image flip</br>
import matplotlib.pyplot as plt</br>
#load image</br>
img=Image.open('cat.jpg')</br>
plt.imshow(img)</br>
plt.show()</br>
#use the flip function</br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)</br>
#save image</br>
flip.save("D:image_flip.jpg")</br>
plt.imshow(flip)</br>
plt.show()</br>
![image](https://user-images.githubusercontent.com/97939934/179968333-d622500f-e5b8-418e-a63c-cdcc69244ba7.png)</br>
</br>
#Importing image class from PIL module</br>
from PIL import Image</br>
import matplotlib.pyplot as plt</br>
#opens a image in RGB mode</br>
im=Image.open('cat.jpg')</br>
</br>
#size of the image in pixels(size of original image)</br>
#this is not mandatory</br>
width, height=im.size</br>
</br>
#cropped image of above dimension</br>
#it will not cange the original image</br>
im1=im.crop((75,100,250,400)) #L-T-R-B</br>

#Shows the image in image viewer</br>
im1.show()</br>
plt.imshow(im1)</br>
plt.show()</br>
![image](https://user-images.githubusercontent.com/97939934/179968405-773ed9f0-ec0f-41c4-a77b-6274b7cd77d6.png)</br>
</br>
</br>
* Program to generate matrix to image</br>
</br>
from PIL import Image</br>
import numpy as np</br>
import matplotlib.pyplot as plt</br>
w, h = 600, 600</br>
data = np.zeros((h, w, 3), dtype=np.uint8)</br>
data[0:100, 0:100] = [255, 0, 0]</br>
data[100:200, 100:200] = [0,255, 0]</br>
data[200:300, 200:300] = [0, 0, 255]</br>
data[300:400, 300:400] = [255, 70, 0]</br>
data[400:500, 400:500] = [255,120, 0]</br>
data[500:600, 500:600] = [ 255, 255, 0]</br>
      #len     width</br>
img = Image.fromarray(data, 'RGB')</br>
plt.imshow(img)</br>
plt.axis("off")</br>
plt.show()</br>
![image](https://user-images.githubusercontent.com/97939934/180202603-b2ea7a81-671b-45b1-8cff-965d15215a68.png)</br>
</br>
</br>
* Program to </br>
</br>
import numpy as np</br>
import matplotlib.pyplot as plt</br>
arr = np.zeros((256,256,3), dtype=np.uint8)</br>
imgsize = arr.shape[:2]</br>
innerColor = (255, 255, 255)</br>
outerColor = (255, 0, 0)</br>
for y in range(imgsize[1]):</br>
    for x in range(imgsize[0]):</br>
        #Find the distance to the center</br>
        distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)</br>
        #Make it on a scale from 0 to 1innerColor</br>
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)</br>
        #Calculate r, g, and b values</br>
        r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)</br>
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)</br>
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)</br>
        # print r, g, b</br>
        arr[y, x] = (int(r), int(g), int(b))</br>

plt.imshow(arr, cmap='gray')</br>
plt.show()</br>

![image](https://user-images.githubusercontent.com/97939934/180202909-88e3f047-158b-4dde-ac36-2c6db25510bf.png)</br>


# 25. Edge Detection</br>
</br>
import cv2</br>
# Read the original image</br>
img = cv2.imread('dog.png')</br>
# Display original image</br>
cv2.imshow('Original', img)</br>
cv2.waitKey(0)</br>
# Convert to graycsale</br>
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)</br>
# Blur the image for better edge detection</br>
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)</br>
# Sobel Edge Detection</br>
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis</br>
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis</br>
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection</br>
# Display Sobel Edge Detection Images</br>
cv2.imshow('Sobel X', sobelx)</br>
cv2.waitKey(0)</br>
cv2.imshow('Sobel Y', sobely)</br>
cv2.waitKey(0)</br>
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)</br>
cv2.waitKey(0)</br>
# Canny Edge Detection</br>
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection</br>
# Display Canny Edge Detection Image</br>
cv2.imshow('Canny Edge Detection', edges)</br>
cv2.waitKey(0)</br>
cv2.destroyAllWindows()</br>
</br>

![image](https://user-images.githubusercontent.com/97939934/187877833-e9575c59-47bf-47fa-a291-6ac32466cfe5.png)</br> 

![image](https://user-images.githubusercontent.com/97939934/187877938-c0b4d43d-b45a-4622-b8c8-00be01117c7b.png)</br>

![image](https://user-images.githubusercontent.com/97939934/187878044-5d811ef1-e1b7-449d-9568-0d3852cf623f.png)</br>

![image](https://user-images.githubusercontent.com/97939934/187878167-f5928c5c-6afa-4602-8190-19e70782e7e3.png)</br>

![image](https://user-images.githubusercontent.com/97939934/187878239-5e1f59c3-409d-4553-97ed-b2f7edbff11d.png)</br>







