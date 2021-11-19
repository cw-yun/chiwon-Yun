'''
# 여러개의 파일로 나누어 놓기 전 전체코드
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import ast

start = time.time() # 시간 측정 시작

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    out = exp_x / sum_exp_x
    return out

def Bias(conv_sum, bias):
    num_channel, num_input, input_width, input_height = conv_sum.shape

    for i in range(num_channel):
            for j in range(num_input):
                for k in range(input_width):
                    for l in range(input_height):
                        conv_sum[i][j][k][l] += bias[j]
    return conv_sum


def Convolution(input, kernel, bias):
    num_channel, num_input, input_width, input_height = input.shape
    num_filter, input_node, kernel_width, kernel_height = kernel.shape
    new_width = input_width - kernel_width + 1
    new_height = input_height - kernel_height + 1

    conv_sum = []

    # input과 filter의 conv 연산
    for i in range(num_filter):
        for j in range(input_node):
            conv = []
            for k in range(new_height):
                for l in range(new_width):
                    conv.append((input[0, j, k:k + kernel_height, l:l + kernel_width] * kernel[i][j]).sum())
            conv_sum.append(conv)

    # input_node가 1인 경우
    if input_node == 1:
        conv_sum = np.array(conv_sum).reshape(num_channel, num_filter, new_width, new_height)

        # add bias
        conv_sum = Bias(conv_sum, bias)

        print('--------------------------------conv layer------------------------------------')
        print(conv_sum.shape)
        #print(conv_sum)

        # conv layer 이미지 출력
        plt.imshow(conv_sum[0][0], cmap = 'Greys')
        plt.show()
        return conv_sum

    # input_node > 1 인 경우(sum을 하는 동작이 필요)
    elif input_node > 1:
        conv_sum = np.array(conv_sum).reshape(num_filter, input_node, new_width * new_height)
        filter_sum = 0
        all_filter_sum = []
        for i in range(num_filter):
            for j in range(new_width * new_height):
                for k in range(input_node):
                    filter_sum += conv_sum[i][k][j]
                all_filter_sum.append(filter_sum)
                filter_sum = 0

        all_filter_sum = np.array(all_filter_sum).reshape(num_channel, num_filter, new_width, new_height)

        # add bias
        all_filter_sum = Bias(all_filter_sum, bias)

        print('--------------------------------conv layer------------------------------------')
        print(all_filter_sum.shape)
        #print(all_filter_sum)

        # conv layer 이미지 출력
        plt.imshow(all_filter_sum[0][0], cmap = 'Greys')
        plt.show()
        return all_filter_sum



def Max_pooling(input,kernel_size):
    num_channel, num_input, input_width, input_height = input.shape
    input = relu(input)
    pool = []
    all_value = []
    for i in range(num_channel):
        for j in range(num_input):
            for k in range(0, input_width, kernel_size[0]):
                 for l in range(0, input_height, kernel_size[1]):
                    for m in range(kernel_size[0]):
                        for n in range(kernel_size[1]):
                            value = input[i][j][k + m][l + n]
                            all_value.append(value)
                    value_max = max(all_value)
                    pool.append(value_max)
                    all_value.clear()  # all_value reset

    new_width = int(input_width/kernel_size[0])
    new_height = int(input_height/kernel_size[1])
    pool = np.array(pool).reshape(num_channel, num_input, new_width, new_height)
    print('--------------------------------pool layer-------------------------------------')
    print(pool.shape)
    #print(pool)

    #pool layer 이미지 출력
    plt.imshow(pool[0][0], cmap = 'Greys')
    plt.show()

    return pool

def Fully_connected(input, fc1_kernel, fc1_kernel_bias, fc2_kernel, fc2_kernel_bias):
    num_channel, num_input, input_width, input_height = input.shape
    fc1_num_output, fc1_input_node = fc1_kernel.shape
    fc2_num_output, fc2_input_node = fc2_kernel.shape

    flatten = input.reshape(num_channel * num_input * input_width * input_height)
    print('---------------------------------flatten---------------------------------------')
    print(flatten.shape)
    #print(flatten)

    fc1_output = []
    for i in range(fc1_num_output):
        fc1_output.append((flatten * fc1_kernel[i]).sum())
    #bias
    for i in range(fc1_num_output):
        fc1_output[i] += fc1_kernel_bias[i]
    print('---------------------------------fc1 layer-------------------------------------')
    fc1_output = np.array(fc1_output)
    print(fc1_output.shape)
    #print(fc1_output)

    fc1_output = relu(fc1_output)

    fc2_output = []
    for i in range(fc2_num_output):
        fc2_output.append((fc1_output * fc2_kernel[i]).sum())

    #bias
    for i in range(fc2_num_output):
        fc2_output[i] += fc2_kernel_bias[i]
    print('---------------------------------fc2 layer-------------------------------------')
    fc2_output = np.array(fc2_output)
    print(fc2_output.shape)
    #print(fc2_output)

    accuracy = softmax(fc2_output)
    print(accuracy)

    return accuracy

test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
test_data = test_data / 255
reshape_test_data = test_data[1:].reshape(1,1,28,28)

# 기존 이미지 출력
reshape_test_data = reshape_test_data.reshape(28, 28)
plt.imshow(reshape_test_data, cmap = 'Greys')
plt.show()

reshape_test_data = test_data[1:].reshape(1,1,28,28)
np.set_printoptions(threshold=np.inf) # 모든 배열의 수 출력

# weight, bias 값 텍스트 파일로부터 불러오기
f = open('weight.txt','r')
lines = f.read() # string 형태로 값 읽기
lines = lines.split('tensor')

for i in range(1, len(lines)):
    lines[i] = ast.literal_eval(lines[i])  # ast 라이브러리를 이용하여 string 형태의 문자열을 list 형태로 변환

# Conv1_kernel weight & bias
Conv1_kernel = lines[1] # shape : (32,1,5,5)
Conv1_kernel_bias = lines[2] # shape : (32)
Conv1_kernel = np.array(Conv1_kernel)
Pool1_kernel_filter_size = (2,2) # 2*2 down sampling

# Conv2_kernel weight & bias
Conv2_kernel = lines[3] # shape : (64,32,5,5)
Conv2_kernel_bias = lines[4] # shape : (64)
Conv2_kernel = np.array(Conv2_kernel)
Pool2_kernel_filter_size = (2,2) # 2*2 down sampling

# fc1_kernel weight & bias
fc1_kernel = lines[5] # shape : (512,1024)
fc1_kernel_bias = lines[6] # shape : (512)
fc1_kernel = np.array(fc1_kernel)

# fc2_kernel weight & bias
fc2_kernel = lines[7] # shape : (10,512)
fc2_kernel_bias = lines[8] # shape : (10)
fc2_kernel = np.array(fc2_kernel)

Conv1 = Convolution(reshape_test_data, Conv1_kernel, Conv1_kernel_bias)
Max_pooling1 = Max_pooling(Conv1, Pool1_kernel_filter_size)
Conv2 = Convolution(Max_pooling1, Conv2_kernel, Conv2_kernel_bias)
Max_pooling2= Max_pooling(Conv2, Pool2_kernel_filter_size)
fc = Fully_connected(Max_pooling2, fc1_kernel, fc1_kernel_bias, fc2_kernel, fc2_kernel_bias)

f.close() # weight 텍스트 파일 닫기

print("걸리는 시간 :", time.time() - start) # 걸리는 시간 출력
'''

# 각 함수를 여러개의 파일로 나누어 놓은 코드
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import ast
import convolution_file
import max_pooling_file
import fc_file

start = time.time() # 시간 측정 시작

test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
test_data = test_data / 255
reshape_test_data = test_data[1:].reshape(1,1,28,28)

# 기존 이미지 출력
reshape_test_data = reshape_test_data.reshape(28, 28)
plt.imshow(reshape_test_data, cmap = 'Greys')
plt.show()

reshape_test_data = test_data[1:].reshape(1,1,28,28)
np.set_printoptions(threshold=np.inf) # 모든 배열의 수 출력

# weight, bias 값 텍스트 파일로부터 불러오기
f = open('weight.txt','r')
lines = f.read() # string 형태로 값 읽기
lines = lines.split('tensor')

for i in range(1, len(lines)):
    lines[i] = ast.literal_eval(lines[i])  # ast 라이브러리를 이용하여 string 형태의 문자열을 list 형태로 변환

# Conv1_kernel weight & bias
Conv1_kernel = lines[1] # shape : (32,1,5,5)
Conv1_kernel_bias = lines[2] # shape : (32)
Conv1_kernel = np.array(Conv1_kernel)
Pool1_kernel_filter_size = (2,2) # 2*2 down sampling

# Conv2_kernel weight & bias
Conv2_kernel = lines[3] # shape : (64,32,5,5)
Conv2_kernel_bias = lines[4] # shape : (64)
Conv2_kernel = np.array(Conv2_kernel)
Pool2_kernel_filter_size = (2,2) # 2*2 down sampling

# fc1_kernel weight & bias
fc1_kernel = lines[5] # shape : (512,1024)
fc1_kernel_bias = lines[6] # shape : (512)
fc1_kernel = np.array(fc1_kernel)

# fc2_kernel weight & bias
fc2_kernel = lines[7] # shape : (10,512)
fc2_kernel_bias = lines[8] # shape : (10)
fc2_kernel = np.array(fc2_kernel)

Conv1 = convolution_file.Convolution(reshape_test_data, Conv1_kernel, Conv1_kernel_bias)
Max_pooling1 = max_pooling_file.Max_pooling(Conv1, Pool1_kernel_filter_size)
Conv2 = convolution_file.Convolution(Max_pooling1, Conv2_kernel, Conv2_kernel_bias)
Max_pooling2= max_pooling_file.Max_pooling(Conv2, Pool2_kernel_filter_size)
fc = fc_file.Fully_connected(Max_pooling2, fc1_kernel, fc1_kernel_bias, fc2_kernel, fc2_kernel_bias)

f.close() # weight 텍스트 파일 닫기

print("걸리는 시간 :", time.time() - start) # 걸리는 시간 출력