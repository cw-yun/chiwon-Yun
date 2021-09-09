import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def nearest(convert_im,modified_width,modifed_height):
    '''modified_width = 112
    modified_height = 112'''
    convert_im.resize((512,512))
    print(convert_im.size)
    convert_im.save('512.JPG')
    return convert_im
'''    
def bilinear():
    
'''
img_name = ('my_image.JPG')
im = Image.open(img_name)

rgb_im = im.convert('RGB')

r, g, b = rgb_im.getpixel((1,1))
print(r,g,b)
pix = np.array(im)

print(im.size)
width, height = im.size
print(width,height)

print(pix)
print(pix.shape)

print(nearest(im,width,height))

#rgb_im.save('after convert to RGB.JPG')
