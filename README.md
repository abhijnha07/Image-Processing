# Image-Processing

1.Program to display GrayScale image using read() and write() operations

pip install opencv-python</br>
import cv2</br>
img=cv2.imread('flower3.jpg',0)</br>
cv2.imshow('Window',img)</br>
cv2.waitKey(0)</br>
cv2.destroyAllWindows()</br>

![image](https://user-images.githubusercontent.com/97939934/173807678-011c312b-e4d6-409f-9366-b4a53721422e.png)



2.Program to perform linear transformation

from PIL import Image</br>
img=Image.open("bfly3.webp")</br>
img=img.rotate(180)</br>
img.show()</br>
cv2.waitKey(0)</br>
cv2.destroyAllWindows()</br>

![image](https://user-images.githubusercontent.com/97939934/175022026-b46db927-a3c0-46b8-bb85-53dee2797ba5.png)



3.Program to display image using matplotlib

import matplotlib.image as mping</br>
import matplotlib.pyplot as plt</br>
img=mping.imread('flower3.jpg')</br>
plt.imshow(img)</br>

![image](https://user-images.githubusercontent.com/97939934/173808767-1e2dc6a1-a6de-427a-9bfb-d3bf33344cfa.png)</br>



4.Program to convert color string to RGB color values

from PIL import ImageColor</br>
img1=ImageColor.getrgb("yellow")</br>
print(img1)</br>
img2=ImageColor.getrgb("red")</br>
print(img2)</br>

OUTPUT:</br>
(255, 255, 0)</br>
(255, 0, 0)</br>



5.Program to create image using colors

from PIL import Image</br>
img=Image.new('RGB',(300,400),(255,0,255))</br>
img.show()</br>

![image](https://user-images.githubusercontent.com/97939934/173809661-33f675b1-d478-48fd-b0c4-81780098c45d.png)</br>



6.Program to visualize images using varoius volor spaces

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

![image](https://user-images.githubusercontent.com/97939934/173810745-d3539ea3-94cb-40cc-9813-6e40e31467a1.png)</br>
![image](https://user-images.githubusercontent.com/97939934/173813167-9424d45b-bb31-4abf-983d-9e17308a5e58.png)</br>
![image](https://user-images.githubusercontent.com/97939934/173813279-d5643224-b653-4fe6-a897-4139e8615e88.png)</br>



7.Program to display image attributes

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



8.Program to convert the original image to gray scale and then to binary

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

![image](https://user-images.githubusercontent.com/97939934/174047484-01102d3e-4909-4f0c-83c0-424792eccdfe.png)</br>
![image](https://user-images.githubusercontent.com/97939934/174047586-f1a7b9d7-9f6a-468c-a3b0-8530b47d28de.png)</br>
![image](https://user-images.githubusercontent.com/97939934/174047676-4c0667bb-7ec8-4773-97f8-243e48496624.png)</br>



9.Program to resize the original image</br>

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



10.Develop a program to readimage using URL.</br>

from skimage import io</br>
import matplotlib.pyplot as plt</br>
url='https://climate.nasa.gov/rails/active_storage/blobs/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBbjRyIiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--0a7f9ec62ad04559ccea084556300e01789e456a/9827327865_98e0f0dc2d_o.jpg'</br>
image=io.imread(url)</br>
plt.imshow(image)</br>
plt.show()</br>
![image](https://user-images.githubusercontent.com/97939934/175019786-b1b4feb0-2990-49ab-9804-f3b3bc86999d.png)</br>



11.Write a program to mask and blur the image</br>

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



12.Write a program to perform arithmatic operations on images.</br>

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

![image](https://user-images.githubusercontent.com/97939934/175020761-bc0131d3-1b12-4a4f-871e-46df65ad9e75.png)</br>
![image](https://user-images.githubusercontent.com/97939934/175020852-952dd631-c406-4f81-bb58-0258d6a599a6.png)</br>
![image](https://user-images.githubusercontent.com/97939934/175020936-9674ffc1-8f29-4fa2-be6f-586ce54fa1c3.png)</br>



13. Develop the program to change the image to different color spaces</br>
 
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



14. Program to create an image using 2D array</br>

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



15.Program to implement Bitwise operations.

import cv2
import matplotlib.pyplot as plt
image1=cv2.imread('b1.jpg')
image2=cv2.imread('b1.jpg')
ax=plt.subplots(figsize=(15,10))
bitwiseAnd=cv2.bitwise_and(image1,image2)
bitwiseOr=cv2.bitwise_or(image1,image2)
bitwiseXor=cv2.bitwise_xor(image1,image2)
bitwiseNot_img1=cv2.bitwise_not(image1)
bitwiseNot_img2=cv2.bitwise_not(image2)
plt.subplot(151)
plt.imshow(bitwiseAnd)
plt.subplot(152)
plt.imshow(bitwiseOr)
plt.subplot(153)
plt.imshow(bitwiseXor)
plt.subplot(154)
plt.imshow(bitwiseNot_img1)
plt.subplot(155)
plt.imshow(bitwiseNot_img2)
cv2.waitKey(0)

![image](https://user-images.githubusercontent.com/97939934/176425366-d3455c3d-0bc5-497b-8bd5-6d7ba573392f.png)



16.Program to implements varoius Blur Techniques

import cv2
import numpy as np
image=cv2.imread('bird1.jpg')
cv2.imshow('Original Image', image)
cv2.waitKey(0)
#Gaussian blur    
Gaussian=cv2.GaussianBlur(image,(7,7),0)
cv2.imshow('Gaussian Blurring', Gaussian)
cv2.waitKey(0)
#Median Blur
median=cv2.medianBlur(image,5)
cv2.imshow('Median Blurring', median)
cv2.waitKey(0)
#Bilateral Blurring
bilateral=cv2.bilateralFilter(image,9,75,75)
cv2.imshow('Bilateral Blurring', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

![image](https://user-images.githubusercontent.com/97939934/176425915-c203002b-ea2c-4493-b857-ad1f432e5b6c.png)
![image](https://user-images.githubusercontent.com/97939934/176426055-804fea45-de0e-4a18-a1f7-a36691f57f80.png)
![image](https://user-images.githubusercontent.com/97939934/176426147-56966a08-d998-4b60-a113-efdcb368424d.png)
![image](https://user-images.githubusercontent.com/97939934/176426237-f19f0d1f-6a66-4c8e-bb24-898541666128.png)



17.Program to perform Image Enhancement

from PIL import Image
from PIL import ImageEnhance
image=Image.open('img2.jpg')
image.show()
enh_bri=ImageEnhance.Brightness(image)
brightness=1.5
image_brightened=enh_bri.enhance(brightness)
image_brightened.show()
enh_col=ImageEnhance.Color(image)
color=1.5
image_colored=enh_col.enhance(color)
image_colored.show()
enh_con=ImageEnhance.Contrast(image)
contrast=1.5
image_contarsted=enh_con.enhance(contrast)
image_contarsted.show()
enh_sha=ImageEnhance.Sharpness(image)
sharpness=3.0
image_sharped=enh_sha.enhance(sharpness)
image_sharped.show()

![image](https://user-images.githubusercontent.com/97939934/176427018-d3c1c315-15f8-4aac-b53a-b2c30f29ee35.png)
![image](https://user-images.githubusercontent.com/97939934/176427080-05413a18-ce62-4f9d-8640-eea0f4bc1fc5.png)
![image](https://user-images.githubusercontent.com/97939934/176427140-d1d0f9fe-89c4-4c1b-aa50-13acaf108e4b.png)
![image](https://user-images.githubusercontent.com/97939934/176427223-9c5b7079-f75c-4fca-ad3e-f9a8edf31244.png)
![image](https://user-images.githubusercontent.com/97939934/176427310-729b6934-27cc-425b-ae31-6e0949b17619.png)



18.Program to perfrom Morphological operations

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image,ImageEnhance
img=cv2.imread('b1.jpg')
ax=plt.subplots(figsize=(20,10))
kernel=np.ones((5,5),np.uint8)
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
erosion=cv2.erode(img,kernel,iterations=1)
dilation=cv2.dilate(img,kernel,iterations=1)
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
plt.subplot(151)
plt.imshow(opening)
plt.subplot(152)
plt.imshow(closing)
plt.subplot(153)
plt.imshow(erosion)
plt.subplot(154)
plt.imshow(dilation)
plt.subplot(155)
plt.imshow(gradient)
cv2.waitKey(0)

![image](https://user-images.githubusercontent.com/97939934/176427695-80a9720d-0c5b-4cd5-a864-838013d3a54d.png)



