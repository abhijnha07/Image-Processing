# Image-Processing
pip install opencv-python
import cv2
img=cv2.imread('flower3.jpg',0)
cv2.imshow('Window',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
![image](https://user-images.githubusercontent.com/97939934/173807678-011c312b-e4d6-409f-9366-b4a53721422e.png)


from PIL import Image
img=Image.open("bfly3.webp")
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
![image](https://user-images.githubusercontent.com/97939934/173808550-9b62329d-327f-45b0-aac2-b522b6b6caec.png)


import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('flower3.jpg')
plt.imshow(img)
![image](https://user-images.githubusercontent.com/97939934/173808767-1e2dc6a1-a6de-427a-9bfb-d3bf33344cfa.png)


from PIL import ImageColor
img1=ImageColor.getrgb("yellow")
print(img1)
img2=ImageColor.getrgb("red")
print(img2)

OUTPUT:
(255, 255, 0)
(255, 0, 0)

from PIL import Image
img=Image.new('RGB',(300,400),(255,0,255))
img.show()
![image](https://user-images.githubusercontent.com/97939934/173809661-33f675b1-d478-48fd-b0c4-81780098c45d.png)


