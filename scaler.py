import numpy as np
from PIL import Image
import cv2
from collections import Counter

def nearest(original_pix,original_width,original_height):
    src = cv2.imread('my_image.JPG', cv2.IMREAD_COLOR)
    modified_width = 512
    modified_height = 512
    
    width_ratio = modified_width / original_width
    height_ratio = modified_height / original_height
    
    new_width_pixel = np.array(range(modified_width)) + 1
    new_height_pixel = np.array(range(modified_height)) + 1
    
    new_width_pixel = new_width_pixel / width_ratio
    new_height_pixel = new_height_pixel / height_ratio
    
    new_width_pixel = np.ceil(new_width_pixel)
    new_height_pixel = np.ceil(new_height_pixel)
    
    width_repeat = np.array(list(Counter(new_width_pixel).values()))
    print(original_pix.shape)
    print(width_repeat)
    
    width_matrix = np.dstack([np.repeat(original_pix[:,i], width_repeat) for i in range(original_pix.shape[1])])[0]
    #print(width_matrix)
    
    #print(new_width_pixel)
    #print(new_height_pixel)
    
    convert_im = cv2.resize(src, (modified_width, modified_height), interpolation = cv2.INTER_NEAREST)
    #print(convert_im.size)
    
    cv2.imwrite('nearest_im.JPG', convert_im)
    cv2.imshow("nearest_im.JPG", convert_im)
    

def bilinear(original_width,original_height):
    src = cv2.imread('my_image.JPG', cv2.IMREAD_COLOR)
    modified_width = 1024
    modified_height = 1024
    
    width_ratio = modified_width / original_width
    height_ratio = modified_height / original_height
    
    convert_im = cv2.resize(src, (modified_width,modified_height), interpolation = cv2.INTER_LINEAR)
    #print(convert_im.size)
    
    
    cv2.imwrite('linear_im.JPG', convert_im)
    cv2.imshow("linear_im.JPG", convert_im)

img_name = ('my_image.JPG')
im = Image.open(img_name)
rgb_im = im.convert('RGB')

r, g, b = rgb_im.getpixel((1,1))
pix = np.array(im)
width, height = im.size

'''
print(r,g,b)
print(im.size)
print(width,height)
print(pix)
print(pix.shape)
'''

nearest(pix,width,height)
bilinear(width,height)

cv2.waitKey()
cv2.destroyAllWindows()
