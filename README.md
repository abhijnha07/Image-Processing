# Image-Processing
pip install opencv-python</br>
import cv2</br>
img=cv2.imread('flower3.jpg',0)</br>
cv2.imshow('Window',img)</br>
cv2.waitKey(0)</br>
cv2.destroyAllWindows()</br>
![image](https://user-images.githubusercontent.com/97939934/173807678-011c312b-e4d6-409f-9366-b4a53721422e.png)


from PIL import Image</br>
img=Image.open("bfly3.webp")</br>
img=img.rotate(180)</br>
img.show()</br>
cv2.waitKey(0)</br>
cv2.destroyAllWindows()</br>
![image](https://user-images.githubusercontent.com/97939934/173808550-9b62329d-327f-45b0-aac2-b522b6b6caec.png)</br>


import matplotlib.image as mping</br>
import matplotlib.pyplot as plt</br>
img=mping.imread('flower3.jpg')</br>
plt.imshow(img)</br>
![image](https://user-images.githubusercontent.com/97939934/173808767-1e2dc6a1-a6de-427a-9bfb-d3bf33344cfa.png)</br>


from PIL import ImageColor</br>
img1=ImageColor.getrgb("yellow")</br>
print(img1)</br>
img2=ImageColor.getrgb("red")</br>
print(img2)</br>

OUTPUT:</br>
(255, 255, 0)</br>
(255, 0, 0)</br>

from PIL import Image</br>
img=Image.new('RGB',(300,400),(255,0,255))</br>
img.show()</br>
![image](https://user-images.githubusercontent.com/97939934/173809661-33f675b1-d478-48fd-b0c4-81780098c45d.png)</br>


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
