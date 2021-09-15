from PIL import Image
import numpy as np
import time

def nearest(original_img,input_size, output_size):
    start = time.time()
    pixel = original_img.load()

    width_scale = output_size[0] / input_size[0]
    height_scale = output_size[1] / input_size[1]

    rgb = {}
    sum_rgb = []
    for j in range(output_size[1]):
        for i in range(output_size[0]):
            position = [i/width_scale, j/height_scale]
            if (position[0] < 0):
                position = (0, position[1])
            if (position[1] < 0):
                position = (position[0], 0)
            if (position[0] > original_img.size[0] - 2):
                position = (original_img.size[0] - 2, position[1])
            if (position[1] > original_img.size[1] - 2):
                position = (position[0], original_img.size[1] - 2)

            left = int(position[0])
            right = int(position[0]) + 1
            top = int(position[1])
            bottom = int(position[1]) + 1

            a = position[0] - int(position[0])
            b = position[1] - int(position[1])

            A = pixel[left, top]
            B = pixel[right, top]
            C = pixel[left, bottom]
            D = pixel[right, bottom]

            if a <= 0.5 and b <= 0.5:
                X = (A[0], A[1], A[2])
            elif a > 0.5 and b <= 0.5:
                X = (B[0], B[1], B[2])
            elif a <= 0.5 and b > 0.5:
                X = (C[0], C[1], C[2])
            elif a > 0.5 and b > 0.5:
                X = (D[0], D[1], D[2])

            rgb[i,j] = X
            sum_rgb += rgb[i,j]
    sum_rgb = np.array(sum_rgb)
    reshape_rgb = np.reshape(sum_rgb, (output_size[1], output_size[0], 3))
    image_rgb = Image.fromarray(reshape_rgb.astype('uint8'))
    print(image_rgb)
    print("time: ",time.time() - start)
    return image_rgb


def bilinear(original_img,input_size,output_size):
    start = time.time()
    pixel = original_img.load()

    width_scale = output_size[0] / input_size[0]
    height_scale = output_size[1] / input_size[1]

    rgb = {}
    sum_rgb = []
    for j in range(output_size[1]):
        for i in range(output_size[0]):
            position = [i/width_scale, j/height_scale]
            if (position[0] < 0):
                position = (0, position[1])
            if (position[1] < 0):
                position = (position[0], 0)
            if (position[0] > original_img.size[0] - 2):
                position = (original_img.size[0] - 2, position[1])
            if (position[1] > original_img.size[1] - 2):
                position = (position[0], original_img.size[1] - 2)

            left = int(position[0])
            right = int(position[0]) + 1
            top = int(position[1])
            bottom = int(position[1]) + 1

            a = position[0] - int(position[0])
            b = position[1] - int(position[1])

            A = pixel[left, top]
            B = pixel[right, top]
            C = pixel[left, bottom]
            D = pixel[right, bottom]
            E = ((1 - a) * A[0] + a * B[0], (1 - a) * A[1] + a * B[1], (1 - a) * A[2] + a * B[2])
            F = ((1 - a) * C[0] + a * D[0], (1 - a) * C[1] + a * D[1], (1 - a) * C[2] + a * D[2])
            X = (int((1 - b) * E[0] + b * F[0]), int((1 - b) * E[1] + b * F[1]), int((1 - b) * E[2] + b * F[2]))
            rgb[i, j] = X
            sum_rgb += rgb[i, j]
    sum_rgb = np.array(sum_rgb)
    reshape_rgb = np.reshape(sum_rgb, (output_size[1], output_size[0], 3))
    image_rgb = Image.fromarray(reshape_rgb.astype('uint8'))
    print(image_rgb)
    print("time: ", time.time() - start)
    return image_rgb

original_img = Image.open("my_image.JPG")
original_size = original_img.size
output_size = (456,638)                # set output_size
nearest_img = nearest(original_img,original_size,output_size)
nearest_img.save("nearest.JPG")
bilinear_img = bilinear(original_img,original_size,output_size)
bilinear_img.save("bilinear.JPG")
