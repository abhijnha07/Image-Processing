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

![image](https://user-images.githubusercontent.com/97939934/174049342-d18e4021-e95a-40fc-8de8-20bb9aedce3c.png)


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



