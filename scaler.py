from PIL import Image
import numpy as np

def nearest(im,coordinate): //함수의 전달인자 사용법이 좀더 직관적이면 좋을것 같습니다. 함수만 보고 입출력에 대해 명확히 알수 있는것이 중요합니다. 
                            //이 함수에서 전달인자는 입력사이즈 출력사이즈 이미지 데이터 등이 되면 좋을것 같아요. 
    pixel = im.load()
    if (coordinate[0] < 0):
        coordinate = (0, coordinate[1])
    if (coordinate[1] < 0):
        coordinate = (coordinate[0], 0)
    if (coordinate[0] > im.size[0] - 2):
        coordinate = (im.size[0] - 2, coordinate[1])
    if (coordinate[1] > im.size[1] - 2):
        coordinate = (coordinate[0], im.size[1] - 2)

    #if (coordinate[0] == int(coordinate[0]) and coordinate[1] == int(coordinate[1])):
    #    return pixel(coordinate)

    left = int(coordinate[0])
    right = int(coordinate[0]) + 1
    top = int(coordinate[1])
    bottom = int(coordinate[1]) + 1

    a = coordinate[0] - int(coordinate[0])
    b = coordinate[1] - int(coordinate[1])
    #print(a)
    #print(b)
    A = pixel[left, top]
    B = pixel[right, top]
    C = pixel[left, bottom]
    D = pixel[right, bottom]

    if a <= 0.5 and b <= 0.5:
        X = (A[0],A[1],A[2])
        #print("a")
    elif a > 0.5 and b <= 0.5:
        X = (B[1],B[2],B[3])
        #print("b")
    elif a <= 0.5 and b > 0.5:
        X = (C[0],C[1],C[2])
        #print("c")
    elif a > 0.5 and b > 0.5:
        X = (D[0],D[1],D[2])
        #print("d")
    return X


def bilinear(im,coordinate,magnitude):
    #print(coordinate[0])
    #print(coordinate[1])
    pixel = im.load()
    coordinate[0] = coordinate[0] * magnitude
    coordinate[1] = coordinate[1] * magnitude
    #print(coordinate[0])
    #print(coordinate[1])
    if(coordinate[0]<0):
        coordinate = (0,coordinate[1])
    if(coordinate[1]<0):
        coordinate = (coordinate[0],0)
    if(coordinate[0]>im.size[0]-2):
        coordinate = (im.size[0]-2, coordinate[1])
    if(coordinate[1]>im.size[1]-2):
        coordinate = (coordinate[0], im.size[1]-2)

    #if(coordinate[0]==int(coordinate[0]) and coordinate[1]==int(coordinate[1])):
    #    return pixel(coordinate)

    left = int(coordinate[0])
    right = int(coordinate[0]) + 1
    top = int(coordinate[1])
    bottom = int(coordinate[1]) + 1

    a = coordinate[0] - int(coordinate[0])
    b = coordinate[1] - int(coordinate[1])

    A = pixel[left,top]
    B = pixel[right,top]
    C = pixel[left,bottom]
    D = pixel[right,bottom]
    E = ((1-a)*A[0]+a*B[0],(1-a)*A[1]+a*B[1],(1-a)*A[2]+a*B[2])
    F = ((1-a)*C[0]+a*D[0],(1-a)*C[1]+a*D[1],(1-a)*C[2]+a*D[2])
    X = (int((1-b)*E[0]+b*F[0]),int((1-b)*E[1]+b*F[1]),int((1-b)*E[2]+b*F[2]))
    return X

def Magnify_nearest(im, magnitude): // 함수 이름이 Magnify(확대)이면 확대만 되는 함수로 오해될것 같아요. 
    modified_size = (int(im.size[0] * magnitude), int(im.size[1] * magnitude))
    modified_rgb = Image.new("RGB", modified_size)
    modified_pixel = modified_rgb.load()
    for i in range(modified_size[0]):
        for j in range(modified_size[1]):
            coordinate = (int(i / magnitude), int(j / magnitude))
            modified_pixel[i,j] = nearest(im, coordinate)

    print(modified_rgb)
    return modified_rgb


def Magnify_bilinear(im, magnitude):
    modified_size = (int(im.size[0] * magnitude), int(im.size[1] * magnitude))
    modified_rgb = Image.new("RGB", modified_size)
    modified_pixel = modified_rgb.load()

    sum_rgb = []
    for i in range(modified_size[0]):
        for j in range(modified_size[1]):
            coordinate = [int(i/magnitude),int(j/magnitude)]
            modified_pixel[i,j] = bilinear(im,coordinate,magnitude)
            sum_rgb += modified_pixel[i,j]
            #print(sum_rgb)
    sum_rgb = np.array(sum_rgb)
    #print(sum_rgb)
    result = np.reshape(sum_rgb,(modified_size[0],modified_size[1],3))
    #result = np.array(sum_rgb.reshpae(modified_size[0],modified_size[1],3))
    #print(result)
    image_rgb = Image.fromarray(sum_rgb)
    print(image_rgb)
    image_rgb = image_rgb.convert("RGB")
    print(image_rgb)
    return image_rgb

im = Image.open("my_image.JPG")
im_nearest = Magnify_nearest(im,1.5)
im_nearest.save("nearest.JPG")
im_bilinear = Magnify_bilinear(im,1.5)
im_bilinear.save("bilinear.JPG")
