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
mask=cv2.inRange(hsv_img,light_orange,dark_orange)
result=cv2.bitwise_and(img,img,mask=mask)
plt.subplot(1,2,1)
plt.imshow(mask,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(result)
plt.show()
![image](https://user-images.githubusercontent.com/97939934/175020098-0507bfa7-589e-4494-bee6-47c6dd3e87d6.png)
light_white=(0,0,200)
dark_white=(145,60,255)
mask_white=cv2.inRange(hsv_img,light_white,dark_white)
result_white=cv2.bitwise_and(img,img,mask=mask_white)
plt.subplot(1,2,1)
plt.imshow(mask_white,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(result_white)
plt.show()
![image](https://user-images.githubusercontent.com/97939934/175020184-1d5e5429-767f-4e79-8a4e-0d00f31d46b1.png)
final_mask=mask+mask_white
final_result=cv2.bitwise_and(img,img,mask=final_mask)
plt.subplot(1,2,1)
plt.imshow(final_mask,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(final_result)
plt.show()
![image](https://user-images.githubusercontent.com/97939934/175020299-7654888a-68d1-4b9c-9413-77864927fdb5.png)
blur=cv2.GaussianBlur(final_result,(7,7),0)
plt.imshow(blur)
plt.show()
![image](https://user-images.githubusercontent.com/97939934/175020508-92a308c8-b349-405b-9c69-318fe2553076.png)


12.Write a program to perform arithmatic operations on images.

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img1=cv2.imread('flower3.jpg')
img2=cv2.imread('leaf1.jpg')
fimg1=img1+img2
plt.imshow(fimg1)
plt.show()
cv2.imwrite('output.jpg',fimg1)
fimg2=img1-img2
plt.imshow(fimg2)
plt.show()
cv2.imwrite('output.jpg',fimg2)
fimg3=img1*img2
plt.imshow(fimg3)
plt.show()
cv2.imwrite('output.jpg',fimg3)
fimg4=img1/img2
plt.imshow(fimg4)
plt.show()
cv2.imwrite('output.jpg',fimg4)
![image](https://user-images.githubusercontent.com/97939934/175020761-bc0131d3-1b12-4a4f-871e-46df65ad9e75.png)
![image](https://user-images.githubusercontent.com/97939934/175020852-952dd631-c406-4f81-bb58-0258d6a599a6.png)
![image](https://user-images.githubusercontent.com/97939934/175020936-9674ffc1-8f29-4fa2-be6f-586ce54fa1c3.png)



13. Develop the program to change the image to different color spaces
 
import cv2
img=cv2.imread('D:\\flower0.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow("GRAY image",gray)
cv2.imshow("HSV image",hsv)
cv2.imshow("LAB image",lab)
cv2.imshow("HLS image",hls)
cv2.imshow("YUV image",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()
![image](https://user-images.githubusercontent.com/97939934/175267640-9245cbcc-f2f1-4d78-b291-9b5b1fed46f1.png)
![image](https://user-images.githubusercontent.com/97939934/175267823-391abf88-72b0-4276-adad-6004c567be65.png)



14. Program to create an image using 2D array

import cv2 as c
import numpy as np
from PIL import Image
array=np.zeros([100,200,3],dtype=np.uint8)
array[:,:100]=[255,130,0]
array[:,100:]=[0,0,255]
img=Image.fromarray(array)
img.save('Image1.png')
img.show()
c.waitKey(0)
![image](https://user-images.githubusercontent.com/97939934/175280445-f94934d7-ddde-46ef-a773-0cc12c7a2b54.png)



