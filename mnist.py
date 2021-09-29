'''
import numpy as np

def Convolution(input,kernel):
    num_input, input_width, input_height = input.shape
    num_filter, kernel_width, kernel_height = kernel.shape

    new_width = input_width - kernel_width + 1
    new_height = input_height - kernel_height + 1
    conv = []
    for i in range(num_input):
        for j in range(num_filter):
            for k in range(new_width):
                for l in range(new_height):
                    conv.append((input[i,k:k + kernel_width, l:l + kernel_height] * kernel[j]).sum())

    global num_all_filter
    num_all_filter *= num_filter
    conv = np.array(conv).reshape(num_all_filter,new_width,new_height)
    print(conv.shape)
    return conv

def Max_pooling(input,kernel_size):
    num_input, input_width, input_height = input.shape

    pool = []
    all_value = []
    for i in range(num_input):
        for j in range(0, input_width, kernel_size[0]):
             for k in range(0, input_height, kernel_size[1]):
                for l in range(kernel_size[0]):
                    for m in range(kernel_size[1]):
                        value = input[i][j + l][k + m]
                        all_value.append(value)
                value_max = max(all_value)
                pool.append(value_max)
                all_value.clear()  # all_value reset


    new_width = int(input_width/kernel_size[0])
    new_height = int(input_height/kernel_size[1])
    pool = np.array(pool).reshape(num_input,new_width,new_height)
    print(pool.shape)
    return pool

def Fully_connected(input):
    num_input, input_width, input_height = input.shape
    flatten = input.reshape(num_input * input_width * input_height)
    print(flatten.shape)


test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
#print("test_data.shape=",test_data.shape)
#test_data = test_data / 255
reshape_test_data = test_data[1:].reshape(1,28,28)
#print(reshape_test_data.shape)

num_all_filter = 1

Conv1_kernel = [[[0.01,0,0.01,0,0.01],[0,0.01,0,0.01,0],[0.01,0,0.01,0,0.01],[0,0.01,0,0.01,0],[0.01,0,0.01,0,0.01]],            # 6 filters, 5*5 kernel
                [[0.011,0,0.011,0,0.011],[0,0.011,0,0.011,0],[0.011,0,0.011,0,0.011],[0,0.011,0,0.011,0],[0.011,0,0.011,0,0.011]],
                [[0.012,0,0.012,0,0.012],[0,0.012,0,0.012,0],[0.012,0,0.012,0,0.012],[0,0.012,0,0.012,0],[0.012,0,0.012,0,0.012]],
                [[0.013,0,0.013,0,0.013],[0,0.013,0,0.013,0],[0.013,0,0.013,0,0.013],[0,0.013,0,0.013,0],[0.013,0,0.013,0,0.013]],
                [[0.014,0,0.014,0,0.014],[0,0.014,0,0.014,0],[0.014,0,0.014,0,0.014],[0,0.014,0,0.014,0],[0.014,0,0.014,0,0.014]],
                [[0.015,0,0.015,0,0.015],[0,0.015,0,0.015,0],[0.015,0,0.015,0,0.015],[0,0.015,0,0.015,0],[0.015,0,0.015,0,0.015]]]
Conv1_kernel = np.array(Conv1_kernel)
Pool1_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

Conv2_kernel = [[[0.01,0,0.01,0,0.01],[0,0.01,0,0.01,0],[0.01,0,0.01,0,0.01],[0,0.01,0,0.01,0],[0.01,0,0.01,0,0.01]],            # 3 filters, 5*5 kernel
                [[0.011,0,0.011,0,0.011],[0,0.011,0,0.011,0],[0.011,0,0.011,0,0.011],[0,0.011,0,0.011,0],[0.011,0,0.011,0,0.011]],
                [[0.012,0,0.012,0,0.012],[0,0.012,0,0.012,0],[0.012,0,0.012,0,0.012],[0,0.012,0,0.012,0],[0.012,0,0.012,0,0.012]]]
Conv2_kernel = np.array(Conv2_kernel)
Pool2_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

Conv1 = Convolution(reshape_test_data,Conv1_kernel)
Max_pooling1 = Max_pooling(Conv1,Pool1_kernel_filter_size)

Conv2 = Convolution(Max_pooling1,Conv2_kernel)
Max_pooling2= Max_pooling(Conv2,Pool2_kernel_filter_size)
Fully_connected = Fully_connected(Max_pooling2)
'''


'''
import numpy as np

def Convolution(input,kernel):
    num_input, input_width, input_height = input.shape
    num_filter, kernel_width, kernel_height = kernel.shape

    new_width = input_width - kernel_width + 1
    new_height = input_height - kernel_height + 1
    conv = []
    for i in range(num_input):
        for j in range(num_filter):
            for k in range(new_width):
                for l in range(new_height):
                    conv.append((input[i,k:k + kernel_width, l:l + kernel_height] * kernel[j]).sum())

    global num_all_filter
    num_all_filter *= num_filter
    conv = np.array(conv).reshape(num_all_filter,new_width,new_height)
    print(conv.shape)
    return conv

def Max_pooling(input,kernel_size):
    num_input, input_width, input_height = input.shape

    pool = []
    all_value = []
    for i in range(num_input):
        for j in range(0, input_width, kernel_size[0]):
             for k in range(0, input_height, kernel_size[1]):
                for l in range(kernel_size[0]):
                    for m in range(kernel_size[1]):
                        value = input[i][j + l][k + m]
                        all_value.append(value)
                value_max = max(all_value)
                pool.append(value_max)
                all_value.clear()  # all_value reset


    new_width = int(input_width/kernel_size[0])
    new_height = int(input_height/kernel_size[1])
    pool = np.array(pool).reshape(num_input,new_width,new_height)
    print(pool.shape)
    return pool

def Fully_connected(input):
    num_input, input_width, input_height = input.shape
    flatten = input.reshape(num_input * input_width * input_height)
    print(flatten.shape)
    print(flatten)


test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
#print("test_data.shape=",test_data.shape)
test_data = test_data / 255
reshape_test_data = test_data[1:].reshape(1,28,28)
#print(reshape_test_data.shape)

num_all_filter = 1

Conv1_kernel = [[[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13]],            # 6 filters, 5*5 kernel
                [[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14]],
                [[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15]],
                [[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18]],
                [[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19]],
                [[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21]]]
Conv1_kernel = np.array(Conv1_kernel)
Pool1_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

Conv2_kernel = [[[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32]],            # 3 filters, 5*5 kernel
                [[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33]],
                [[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35]]]
Conv2_kernel = np.array(Conv2_kernel)
Pool2_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

Conv1 = Convolution(reshape_test_data,Conv1_kernel)
Max_pooling1 = Max_pooling(Conv1,Pool1_kernel_filter_size)

Conv2 = Convolution(Max_pooling1,Conv2_kernel)
Max_pooling2= Max_pooling(Conv2,Pool2_kernel_filter_size)
Fully_connected = Fully_connected(Max_pooling2)
'''



'''
import numpy as np

def relu(x):
    return np.maximum(0,x)

def Convolution(input,kernel):
    num_input, input_width, input_height = input.shape
    num_filter, kernel_width, kernel_height = kernel.shape

    new_width = input_width - kernel_width + 1
    new_height = input_height - kernel_height + 1
    conv = []
    for i in range(num_input):
        for j in range(num_filter):
            for k in range(new_width):
                for l in range(new_height):
                    conv.append((input[i,k:k + kernel_width, l:l + kernel_height] * kernel[j]).sum())

    global num_all_filter
    num_all_filter *= num_filter
    conv = np.array(conv).reshape(num_all_filter,new_width,new_height)
    conv = relu(conv)       # activation function
    print(conv.shape)
    return conv

def Max_pooling(input,kernel_size):
    num_input, input_width, input_height = input.shape

    pool = []
    all_value = []
    for i in range(num_input):
        for j in range(0, input_width, kernel_size[0]):
             for k in range(0, input_height, kernel_size[1]):
                for l in range(kernel_size[0]):
                    for m in range(kernel_size[1]):
                        value = input[i][j + l][k + m]
                        all_value.append(value)
                value_max = max(all_value)
                pool.append(value_max)
                all_value.clear()  # all_value reset


    new_width = int(input_width/kernel_size[0])
    new_height = int(input_height/kernel_size[1])
    pool = np.array(pool).reshape(num_input,new_width,new_height)
    print(pool.shape)
    return pool

def Fully_connected(input):
    num_input, input_width, input_height = input.shape
    flatten = input.reshape(num_input * input_width * input_height)
    print(flatten.shape)
    print(flatten)


test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
#print("test_data.shape=",test_data.shape)
test_data = test_data / 255
reshape_test_data = test_data[1:].reshape(1,28,28)
#print(reshape_test_data.shape)

num_all_filter = 1

Conv1_kernel = np.random.rand(6,5,5)  # 6 filters, 5*5 kernel(0~1 normal distribution)
Pool1_kernel_filter_size = (2,2)     # 2*2 down sampling

Conv2_kernel = np.random.rand(3,5,5)  # 3 filters, 5*5 kernel(0~1 normal distribution)
Pool2_kernel_filter_size = (2,2)     # 2*2 down sampling

Conv1 = Convolution(reshape_test_data,Conv1_kernel)
Max_pooling1 = Max_pooling(Conv1,Pool1_kernel_filter_size)

Conv2 = Convolution(Max_pooling1,Conv2_kernel)
Max_pooling2= Max_pooling(Conv2,Pool2_kernel_filter_size)
Fully_connected = Fully_connected(Max_pooling2)
'''


'''
import numpy as np

def Convolution(input,kernel):
    input_width, input_height, num_input = input.shape
    num_filter, kernel_width, kernel_height = kernel.shape

    new_width = input_width - kernel_width + 1
    new_height = input_height - kernel_height + 1
    conv = []
    for i in range(num_input):
        for j in range(num_filter):
            for k in range(new_width):
                for l in range(new_height):
                    conv.append((input[k:k + kernel_width, l:l + kernel_height, i] * kernel[j]).sum())

    global num_all_filter
    num_all_filter *= num_filter
    conv = np.array(conv).reshape(new_width, new_height, num_all_filter)
    print(conv.shape)
    return conv

def Max_pooling(input,kernel_size):
    input_width, input_height, num_input = input.shape

    pool = []
    all_value = []
    for i in range(num_input):
        for j in range(0, input_width, kernel_size[0]):
             for k in range(0, input_height, kernel_size[1]):
                for l in range(kernel_size[0]):
                    for m in range(kernel_size[1]):
                        value = input[j + l][k + m][i]
                        all_value.append(value)
                value_max = max(all_value)
                pool.append(value_max)
                all_value.clear()  # all_value reset


    new_width = int(input_width/kernel_size[0])
    new_height = int(input_height/kernel_size[1])
    pool = np.array(pool).reshape(new_width, new_height, num_input)
    print(pool.shape)
    return pool

def Fully_connected(input):
    input_width, input_height, num_input = input.shape
    flatten = input.reshape(num_input * input_width * input_height)
    print(flatten.shape)
    #print(flatten)


test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
#print("test_data.shape=",test_data.shape)
test_data = test_data / 255
reshape_test_data = test_data[1:].reshape(28,28,1)
#print(reshape_test_data.shape)

num_all_filter = 1

Conv1_kernel = [[[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13]],            # 6 filters, 5*5 kernel
                [[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14]],
                [[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15]],
                [[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18]],
                [[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19]],
                [[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21]]]
Conv1_kernel = np.array(Conv1_kernel)
Pool1_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

Conv2_kernel = [[[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32]],            # 3 filters, 5*5 kernel
                [[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33]],
                [[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35]]]
Conv2_kernel = np.array(Conv2_kernel)
Pool2_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

Conv1 = Convolution(reshape_test_data,Conv1_kernel)
Max_pooling1 = Max_pooling(Conv1,Pool1_kernel_filter_size)

Conv2 = Convolution(Max_pooling1,Conv2_kernel)
Max_pooling2= Max_pooling(Conv2,Pool2_kernel_filter_size)
Fully_connected = Fully_connected(Max_pooling2)
'''


'''
import numpy as np                     # 컨볼루션 층에서 input을 각각 다 합쳐서 계산하는 방식

def Convolution(input,kernel):
    input_width, input_height, num_input = input.shape
    num_filter, kernel_width, kernel_height = kernel.shape

    sum_input = 0
    new_input = []

    for i in range(input_width):
        for j in range(input_height):
            for k in range(num_input):
                sum_input += input[i,j,k]
            new_input.append(sum_input)
            sum_input = 0                   # sum_input reset
    new_input = np.array(new_input).reshape(input_width, input_height, 1)
    new_width = input_width - kernel_width + 1
    new_height = input_height - kernel_height + 1

    conv = []
    for i in range(num_filter):
        for j in range(new_width):
            for k in range(new_height):
                conv.append((new_input[j:j + kernel_width, k:k + kernel_height, 0] * kernel[i]).sum())

    conv = np.array(conv).reshape(new_width, new_height, num_filter)
    print(conv.shape)
    return conv

def Max_pooling(input,kernel_size):
    input_width, input_height, num_input = input.shape

    pool = []
    all_value = []
    for i in range(num_input):
        for j in range(0, input_width, kernel_size[0]):
             for k in range(0, input_height, kernel_size[1]):
                for l in range(kernel_size[0]):
                    for m in range(kernel_size[1]):
                        value = input[j + l][k + m][i]
                        all_value.append(value)
                value_max = max(all_value)
                pool.append(value_max)
                all_value.clear()  # all_value reset


    new_width = int(input_width/kernel_size[0])
    new_height = int(input_height/kernel_size[1])
    pool = np.array(pool).reshape(new_width, new_height, num_input)
    print(pool.shape)
    return pool

def Fully_connected(input):
    input_width, input_height, num_input = input.shape
    flatten = input.reshape(num_input * input_width * input_height)
    print(flatten.shape)
    #print(flatten)

test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
#print("test_data.shape=",test_data.shape)
test_data = test_data / 255
reshape_test_data = test_data[1:].reshape(28,28,1)
#print(reshape_test_data.shape)

Conv1_kernel = [[[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13]],            # 6 filters, 5*5 kernel
                [[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14]],
                [[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15]],
                [[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18]],
                [[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19]],
                [[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21]]]
Conv1_kernel = np.array(Conv1_kernel)
Pool1_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

Conv2_kernel = [[[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32]],            # 3 filters, 5*5 kernel
                [[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33]],
                [[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35]]]
Conv2_kernel = np.array(Conv2_kernel)
Pool2_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

Conv1 = Convolution(reshape_test_data,Conv1_kernel)
Max_pooling1 = Max_pooling(Conv1,Pool1_kernel_filter_size)

Conv2 = Convolution(Max_pooling1,Conv2_kernel)
Max_pooling2= Max_pooling(Conv2,Pool2_kernel_filter_size)
Fully_connected = Fully_connected(Max_pooling2)
'''


'''
Conv1_kernel = [[[[ 1.67094380e-01, -1.77180529e-01, -4.71551687e-01, -1.93822116e-01, 1.36287123e-01,  2.38348156e-01]],
                 [[ 8.97379816e-02, -1.27359137e-01, -2.69971073e-01,  2.17470288e-01, -2.61439458e-02,  3.47288638e-01]],
                 [[ 2.45294735e-01,  1.32189933e-02, -2.18005374e-01,  1.35951549e-01, -3.72244745e-01,  3.98036987e-01]],
                 [[ 1.86809063e-01, -7.85414726e-02,  2.95821894e-02,  8.97610709e-02, -3.34434450e-01,  4.07203585e-01]],
                 [[-2.29761787e-02, -1.90833256e-01,  1.56042278e-02,  4.61040959e-02, -1.83083624e-01,  2.40346506e-01]]],
                [[[ 1.91005453e-01,  3.47328633e-02, -9.84267667e-02, -1.85712203e-01, 2.26044863e-01, -1.21728756e-01]],
                 [[ 2.48038441e-01, -2.42267415e-01, -3.07969660e-01,  5.78389578e-02, 1.84232712e-01,  3.32848907e-01]],
                 [[ 6.61106557e-02, -8.51367861e-02, -1.57826439e-01,  2.47479334e-01, 7.35986829e-02,  2.48634964e-01]],
                 [[-7.57155716e-02, -9.95359942e-02, -1.51629463e-01,  3.23129773e-01, -3.04444045e-01,  3.38968188e-01]],
                 [[-1.22465983e-01, -4.02057976e-01,  7.94960943e-04,  3.45858961e-01, -4.68735099e-01,  1.68319032e-01]]],
                [[[ 6.09288812e-02,  2.98400968e-01, -2.16035664e-01, -2.79128909e-01, -2.49070600e-02, -5.55095196e-01]],
                [[ 2.45503932e-01,  6.62921220e-02, -2.05475941e-01, -2.87136555e-01, 3.86290103e-01,  5.74849062e-02]],
                [[ 1.24202199e-01, -2.17195358e-02, -1.82127833e-01, -1.86002463e-01, 2.80802459e-01,  1.49210356e-02]],
                [[-1.29296169e-01, -6.21485375e-02, -1.06248567e-02,  2.34563619e-01, 3.17129672e-01,  1.63708538e-01]],
                [[-2.37110034e-01,  1.65250450e-01,  3.19648124e-02,  9.87904146e-02, 8.54154155e-02,  6.20286018e-02]]],
                [[[ 2.64266610e-01,  2.38384545e-01, -6.21965341e-02, -4.87131119e-01, 1.07345566e-01, -8.21851790e-01]],
                [[ 2.18335569e-01,  2.59376556e-01,  1.54708410e-02, -1.79737866e-01, 1.16388410e-01, -5.31684220e-01]],
                [[ 1.29895315e-01,  1.73845261e-01,  2.65149057e-01, -1.65988952e-01, 2.12555379e-01, -6.07524455e-01]],
                [[-1.09471381e-01,  3.88127983e-01,  1.75194293e-01,  2.43370607e-01, 2.49128103e-01, -5.63060939e-01]],
                [[-3.52162391e-01,  3.19904983e-01,  3.27137887e-01,  1.64880484e-01, 1.91338569e-01, -1.67967111e-01]]],
                [[[ 6.02521934e-02, -1.52366143e-02,  8.45037699e-02, -3.29230428e-01, -2.18086272e-01, -1.61821306e-01]],
                [[ 2.27068260e-01,  1.35485604e-01,  2.89995044e-01, -1.97986871e-01, 1.32868424e-01, -3.01568419e-01]],
                [[ 1.42230287e-01,  1.26459539e-01,  1.35385081e-01, -1.79014996e-01, 8.89067166e-03, -2.99682975e-01]],
                [[ 1.11535482e-01,  7.12459981e-02,  1.85344905e-01,  1.14716344e-01, 6.82652965e-02, -1.63496628e-01]],
                [[ 1.74192488e-02, -2.11645570e-02,  3.04915994e-01,  2.78766811e-01, 2.24847928e-01, -2.62693048e-01]]]]

Conv2_kernel = [[[[ 4.46148030e-03, -2.43060946e-01,  1.82859078e-01],
                  [ 5.73413447e-02, -1.79924056e-01, -1.34274155e-01],
                  [-1.11200668e-01,  1.93545580e-01, -5.03118383e-03],
                  [ 1.94545552e-01,  1.04310684e-01,  5.04743196e-02],
                  [ 1.61035210e-01, -2.22076312e-01, -1.83698922e-01],
                  [ 1.09776214e-01, -6.30918369e-02, -1.09144427e-01]],
                 [[ 1.66212052e-01, -9.49484855e-02,  2.05616966e-01],
                  [ 1.19083099e-01,  2.60210992e-03,  3.32018249e-02],
                  [-3.72494161e-02,  9.93574411e-02,  4.38530818e-02],
                  [ 3.53315592e-01,  2.35935867e-01, -1.80826746e-02],
                  [ 1.75157294e-01,  7.85644278e-02, -2.57276833e-01],
                  [-1.04187012e-01, -1.14900813e-01, -1.69311598e-01]],
                 [[-3.60860233e-03,  7.67827183e-02,  3.21077257e-01],
                  [-4.30341735e-02, 2.28030458e-02,-1.24694742e-01],
                  [-2.73440689e-01, 1.60807729e-01,-1.80979595e-01],
                  [ 4.56692785e-01,  4.73841615e-02,  1.71454191e-01],
                  [ 2.81815767e-01,  1.43617451e-01, -1.96128637e-01],
                  [-1.90690622e-01, -2.81605422e-01, -4.41021137e-02]],
                 [[ 1.94376633e-01, -4.90035489e-02,  1.55250564e-01],
                  [-5.97173981e-02,  1.79853737e-02, -3.07055146e-01],
                  [-4.78641093e-02, -6.51851892e-02, -1.55271634e-01],
                  [ 2.67386168e-01, -3.70349400e-02,  1.19192161e-01],
                  [ 1.40169978e-01, -8.76651630e-02, -2.20318630e-01],
                  [-6.97066784e-01, -2.05059871e-02, -7.47444779e-02]],
                 [[ 1.65673018e-01,  3.31976041e-02, -9.33685824e-02],
                  [ 9.76644829e-03, -8.88296682e-03, -1.78523988e-01],
                  [ 1.09903105e-01, -2.97389686e-01, 1.63721328e-03],
                  [ 8.90138187e-03, -2.54530102e-01,  2.09646225e-02],
                  [ 7.52680078e-02,  9.90150794e-02, -1.28742591e-01],
                  [-3.59054297e-01,  2.79829443e-01, -1.00077577e-01]]],
                [[[-1.29625782e-01, -1.16864361e-01,  2.43602201e-01],
                  [ 1.52480423e-01,  7.45080933e-02, -1.17025144e-01],
                  [ 9.48696211e-02, -8.58374126e-03,  2.17507482e-02],
                  [-7.72828087e-02, -1.41286120e-01,  6.44770712e-02],
                  [ 1.51869044e-01, -8.63722712e-02, -2.28641585e-01],
                  [ 1.52700514e-01, -2.26914227e-01, -1.62951291e-01]],
                 [[-1.33052662e-01,  8.91523156e-03,  1.82868615e-01],
                  [ 2.25476727e-01,  1.91594884e-01, -4.13371287e-02],
                  [ 6.53660744e-02,  1.66390345e-01, -7.79683962e-02],
                  [ 6.98596658e-03, -9.50679556e-02, -1.01213962e-01],
                  [ 2.53964067e-01, -4.62014824e-02, -2.57423490e-01],
                  [ 7.45303780e-02, -2.23321736e-01,  1.22679219e-01]],
                 [[-1.95911713e-02,  1.70287654e-01,  3.81079875e-02],
                  [ 5.45673482e-02,  1.89345464e-01, -3.34087640e-01],
                  [-3.57262231e-02, -1.17123604e-01, -1.00682363e-01],
                  [ 1.59529313e-01,  5.27389646e-02,  1.70735374e-01],
                  [ 5.71238063e-02, -1.00585759e-01, -1.31461665e-01],
                  [-3.29621971e-01, -3.67517620e-02,  3.61060798e-01]],
                 [[ 1.86043113e-01,  7.06016943e-02, -1.37504622e-01],
                  [-1.15026623e-01, -8.08171406e-02, -2.01290429e-01],
                  [-1.80505171e-01, -1.00968689e-01, -2.57065922e-01],
                  [ 2.96369195e-01, -2.64048707e-02,  4.47448641e-02],
                  [ 2.83713996e-01,  1.68162256e-01, -1.12244435e-01],
                  [-4.45447177e-01,  1.39168054e-01, -1.52649119e-01]],
                 [[ 2.10947499e-01, -1.25490397e-01, -3.24189782e-01],
                  [-8.20837244e-02, -7.29732066e-02, -1.85146675e-01],
                  [-1.46288827e-01, -6.91832080e-02,  1.85829520e-01],
                  [ 1.93143144e-01, -1.15489855e-01,  6.80787265e-02],
                  [ 1.83288798e-01,  3.85825410e-02,  7.78506929e-03],
                  [-3.44356894e-01,  1.64550334e-01, -2.28331745e-01]]],
                [[[ 6.02760985e-02,  3.27797271e-02,  1.56385407e-01],
                  [ 1.52392656e-01, -4.88352701e-02, -4.10895161e-02],
                  [ 1.08795747e-01, -1.41858324e-01,  2.36514937e-02],
                  [-2.94398695e-01, -1.63445443e-01, -1.08941413e-01],
                  [ 1.83667898e-01, -8.91597420e-02, -1.26203120e-01],
                  [ 2.21730471e-02, -2.88801473e-02, -1.38151348e-01]],
                 [[ 7.43895322e-02, -9.31640640e-02,  1.92009464e-01],
                  [ 3.30580212e-02,  1.67621151e-01,  1.03326827e-01],
                  [ 1.45272121e-01, -1.20884500e-01,  8.66819918e-02],
                  [-3.03120553e-01, -1.68276295e-01,  1.42453536e-01],
                  [-1.03640564e-01,  5.45207076e-02,  2.38285647e-04],
                  [-2.91135192e-01,  7.13689476e-02,  2.20958859e-01]],
                 [[ 1.76585987e-01, -1.81971397e-02,  1.19804353e-01],
                  [-2.00220361e-01,  1.06862605e-01, -2.99832448e-02],
                  [ 1.07068472e-01, -1.09028660e-01, -1.42807901e-01],
                  [ 5.12155779e-02,  1.54463902e-01, -1.63206279e-01],
                  [-5.36084250e-02,  2.96080709e-01, -5.75776361e-02],
                  [-2.26433888e-01,  2.46767059e-01,  7.96443820e-02]],
                 [[-1.59939658e-02, -2.28865016e-02, -2.26515904e-01],
                  [-5.22559062e-02,  3.40456128e-01, -2.26670265e-01],
                  [-2.08191216e-01, -2.64145937e-02, -2.01001644e-01],
                  [ 2.46572062e-01,  6.09087991e-03,  1.86256301e-02],
                  [-4.72981185e-02,  2.47957766e-01, -1.65833145e-01],
                  [-2.73029149e-01,  3.71114820e-01, -9.85869020e-02]],
                 [[ 1.57899991e-01,  1.64293610e-02, -1.10490508e-01],
                  [-1.22783020e-01,  3.75140458e-01,  1.42274434e-02],
                  [-3.32721680e-01,  5.29803149e-02, -1.12240158e-01],
                  [ 4.58585352e-01, -1.65844932e-01,  1.91291213e-01],
                  [ 1.97771683e-01,  2.43736207e-01, -2.54018214e-02],
                  [-3.91964525e-01,  4.67040241e-01, -2.44016305e-01]]],
                [[[ 1.45044401e-01,  8.88259783e-02,  2.22703204e-01],
                  [ 1.89438015e-01,  1.14392020e-01,  1.66710421e-01],
                  [ 1.42902926e-01,  6.10344298e-02, -1.43264991e-03],
                  [-1.02467267e-02, -6.87396480e-03, -1.00042960e-02],
                  [-1.03194959e-01,  1.71727203e-02, -1.94432199e-04],
                  [ 5.30836917e-02,  1.77650079e-02, -1.96640998e-01]],
                 [[ 7.59854913e-02, -1.36784781e-02,  6.58905208e-02],
                  [-7.41148964e-02,  9.75294691e-03,  1.34717479e-01],
                  [-9.34582502e-02, -1.29915223e-01, -3.98084819e-02],
                  [ 2.28281524e-02, -2.40060300e-01,  7.63384849e-02],
                  [-2.51198024e-01,  7.19136968e-02,  3.39888260e-02],
                  [ 1.09661847e-01,  1.38019159e-01, -1.56892434e-01]],
                 [[-5.25103509e-02, -2.46554062e-01,  3.94752808e-02],
                  [-2.05721796e-01,  3.15680839e-02,  1.22758076e-01],
                  [-1.62131369e-01, -3.06537122e-01,  3.39966826e-02],
                  [ 1.74265549e-01, -7.05152797e-03, -5.80022931e-02],
                  [-6.99093193e-02,  2.52193272e-01,  4.99270968e-02],
                  [-2.06616253e-01,  4.84431446e-01, -2.95978367e-01]],
                 [[ 2.01963410e-01, -9.21719819e-02,  8.73498619e-04],
                  [ 7.58055151e-02,  5.26717678e-03, -8.90280306e-02],
                  [ 5.14510423e-02, -2.15486318e-01,  6.33772314e-02],
                  [ 2.34939083e-01,  1.96062446e-01, -8.18915144e-02],
                  [-1.28485039e-01, -1.83798391e-02,  2.48003393e-01],
                  [-3.66669804e-01,  4.63300794e-01, -2.89995372e-01]],
                 [[ 2.27360606e-01,  1.79621410e-02,  6.30379841e-02],
                  [ 1.35021821e-01,  9.80336815e-02,  1.02929279e-01],
                  [-1.02012768e-01, -5.26271202e-02,  2.12062821e-01],
                  [ 1.29118726e-01,  1.31497100e-01 , 1.80518001e-01],
                  [ 2.75496572e-01,  1.24823213e-01,  1.12401508e-01],
                  [-2.15923175e-01,  5.24682343e-01, -1.83883440e-02]]],
                [[[-2.49553677e-02, -2.41490945e-01,  5.74973412e-02],
                  [-8.41290727e-02,  2.35281765e-01,  3.16370465e-02],
                  [-1.04939409e-01, -1.52145937e-01,  1.23863772e-01],
                  [ 6.01596832e-02,  1.82747915e-01, -6.35440424e-02],
                  [-1.54443666e-01,  2.97095776e-01,  7.88350478e-02],
                  [ 3.32017303e-01,  1.34418994e-01, -9.99353826e-03]],
                 [[-1.87003255e-01, -2.63412982e-01,  9.68348235e-02],
                  [ 1.84518192e-02,  2.79142261e-01,  1.84531540e-01],
                  [ 2.23084330e-03,  2.61985548e-02,  2.03359574e-01],
                  [-1.70044508e-02, -9.67519134e-02, -7.77086392e-02],
                  [-3.99361700e-02,  3.04613382e-01,  2.61705875e-01],
                  [ 8.29325765e-02,  3.33145469e-01,  5.12788594e-02]],
                 [[ 4.83596092e-03, -6.12523705e-02,  2.36663036e-02],
                  [-1.47255406e-01,  2.55434662e-02,  1.35068417e-01],
                  [-2.36879632e-01,  8.61347653e-03, -7.36168073e-03],
                  [ 1.34673655e-01, -3.11976969e-02,  2.77323071e-02],
                  [-3.68976370e-02,  2.38494650e-01,  3.52538556e-01],
                  [-1.06586613e-01,  4.50379074e-01,  1.36053771e-01]],
                 [[ 1.55939251e-01, -1.34711191e-01, -1.01322848e-02],
                  [-2.78911889e-02,  1.64967149e-01,  2.18779057e-01],
                  [-1.29998531e-02, -1.29886180e-01,  2.92970896e-01],
                  [-3.63288224e-02,  1.46652618e-02,  1.16084881e-01],
                  [-7.92094618e-02, -1.30063429e-01,  3.39011014e-01],
                  [-2.15137824e-01,  4.46359932e-01,  3.55196655e-01]],
                 [[ 1.12604082e-01, -1.34613067e-01, -5.26728593e-02],
                  [-6.50925329e-04,  1.34872524e-02,  2.66594529e-01],
                  [ 1.08478405e-01, -1.06381148e-01,  4.47653562e-01],
                  [ 1.24460444e-01, -6.48147911e-02,  7.65434355e-02],
                  [ 2.63863616e-02, -2.06157222e-01, -2.69554730e-04],
                  [ 2.90843427e-01,  1.78602770e-01,  7.68283382e-02]]]]
'''

'''
import numpy as np

def Convolution(input,kernel):
    input_width, input_height, num_input = input.shape
    num_filter, kernel_width, kernel_height = kernel.shape

    new_width = input_width - kernel_width + 1
    new_height = input_height - kernel_height + 1

    conv_sum = []
    for i in range(num_filter):
        for j in range(num_input):
            conv = []
            for k in range(new_width):
                for l in range(new_height):
                    conv.append((input[k:k + kernel_width, l:l + kernel_height, j] * kernel[i]).sum())
        conv_sum.append(conv)


    #계산한 값과 맞지 않는다면 for문을 이용하여 conv_sum의 6개의 리스트에서 각각의 합을 더한 1개의 리스트로 바꾸기
    conv_sum = np.array(conv_sum).reshape(new_width, new_height, num_filter)
    print(conv_sum.shape)
    return conv_sum


def Max_pooling(input,kernel_size):
    input_width, input_height, num_input = input.shape

    pool = []
    all_value = []
    for i in range(num_input):
        for j in range(0, input_width, kernel_size[0]):
             for k in range(0, input_height, kernel_size[1]):
                for l in range(kernel_size[0]):
                    for m in range(kernel_size[1]):
                        value = input[j + l][k + m][i]
                        all_value.append(value)
                value_max = max(all_value)
                pool.append(value_max)
                all_value.clear()  # all_value reset


    new_width = int(input_width/kernel_size[0])
    new_height = int(input_height/kernel_size[1])
    pool = np.array(pool).reshape(new_width, new_height, num_input)
    print(pool.shape)
    return pool

def Fully_connected(input):
    input_width, input_height, num_input = input.shape
    flatten = input.reshape(num_input * input_width * input_height)
    print(flatten.shape)
    #print(flatten)


test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
#print("test_data.shape=",test_data.shape)
test_data = test_data / 255
reshape_test_data = test_data[1:].reshape(28,28,1)
#print(reshape_test_data.shape)

Conv1_kernel = [[[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13],[0.13,0.13,0.13,0.13,0.13]],            # 6 filters, 5*5 kernel
                [[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14],[0.14,0.14,0.14,0.14,0.14]],
                [[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15],[0.15,0.15,0.15,0.15,0.15]],
                [[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18],[0.18,0.18,0.18,0.18,0.18]],
                [[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19],[0.19,0.19,0.19,0.19,0.19]],
                [[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21],[0.21,0.21,0.21,0.21,0.21]]]
Conv1_kernel = np.array(Conv1_kernel)
Pool1_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

Conv2_kernel = [[[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32],[0.32,0.32,0.32,0.32,0.32]],            # 3 filters, 5*5 kernel
                [[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33],[0.33,0.33,0.33,0.33,0.33]],
                [[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35],[0.35,0.35,0.35,0.35,0.35]]]
Conv2_kernel = np.array(Conv2_kernel)
Pool2_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

Conv1 = Convolution(reshape_test_data,Conv1_kernel)
Max_pooling1 = Max_pooling(Conv1,Pool1_kernel_filter_size)

Conv2 = Convolution(Max_pooling1,Conv2_kernel)
Max_pooling2= Max_pooling(Conv2,Pool2_kernel_filter_size)
Fully_connected = Fully_connected(Max_pooling2)
'''

import numpy as np

def Convolution(input,kernel):
    input_width, input_height, num_input, num_channel = input.shape
    kernel_width, kernel_height, input_node, num_filter = kernel.shape
    new_width = input_width - kernel_width + 1
    new_height = input_height - kernel_height + 1


    new_kernel = []
    for i in range(num_filter):
        for j in range(input_node):
            for k in range(kernel_width):
                for l in range(kernel_height):
                    new_kernel.append(kernel[k][l][j][i])
    new_kernel = np.array(new_kernel).reshape(num_filter, input_node, kernel_height, kernel_width)

    '''
    print(new_kernel[0][0])
    print(new_kernel[0][0].shape)
    print(new_kernel[0])
    print(new_kernel[0].shape)
    '''

    conv_sum = []
    for i in range(num_filter):
        for j in range(num_input):
            conv = []
            for k in range(new_width):
                for l in range(new_height):
                    conv.append((input[k:k + kernel_width, l:l + kernel_height, j] * new_kernel[i]).sum())
        conv_sum.append(conv)

    conv_sum = np.array(conv_sum).reshape(new_width, new_height, num_filter, num_channel)
    print(conv_sum.shape)
    return conv_sum


def Max_pooling(input,kernel_size):
    input_width, input_height, num_input, num_channel = input.shape

    pool = []
    all_value = []
    for i in range(num_input):
        for j in range(0, input_width, kernel_size[0]):
             for k in range(0, input_height, kernel_size[1]):
                for l in range(kernel_size[0]):
                    for m in range(kernel_size[1]):
                        value = input[j + l][k + m][i]
                        all_value.append(value)
                value_max = max(all_value)
                pool.append(value_max)
                all_value.clear()  # all_value reset


    new_width = int(input_width/kernel_size[0])
    new_height = int(input_height/kernel_size[1])
    pool = np.array(pool).reshape(new_width, new_height, num_input)
    print(pool.shape)
    return pool

def Fully_connected(input):
    input_width, input_height, num_input = input.shape
    flatten = input.reshape(num_input * input_width * input_height)
    print(flatten.shape)
    #print(flatten)


test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
#print("test_data.shape=",test_data.shape)
test_data = test_data / 255
reshape_test_data = test_data[1:].reshape(28,28,1,1)
#print(reshape_test_data.shape)

Conv1_kernel = [[[[ 1.67094380e-01, -1.77180529e-01, -4.71551687e-01, -1.93822116e-01, 1.36287123e-01,  2.38348156e-01]],   # (5,5,1,6)
                 [[ 8.97379816e-02, -1.27359137e-01, -2.69971073e-01,  2.17470288e-01, -2.61439458e-02,  3.47288638e-01]],
                 [[ 2.45294735e-01,  1.32189933e-02, -2.18005374e-01,  1.35951549e-01, -3.72244745e-01,  3.98036987e-01]],
                 [[ 1.86809063e-01, -7.85414726e-02,  2.95821894e-02,  8.97610709e-02, -3.34434450e-01,  4.07203585e-01]],
                 [[-2.29761787e-02, -1.90833256e-01,  1.56042278e-02,  4.61040959e-02, -1.83083624e-01,  2.40346506e-01]]],
                [[[ 1.91005453e-01,  3.47328633e-02, -9.84267667e-02, -1.85712203e-01, 2.26044863e-01, -1.21728756e-01]],
                 [[ 2.48038441e-01, -2.42267415e-01, -3.07969660e-01,  5.78389578e-02, 1.84232712e-01,  3.32848907e-01]],
                 [[ 6.61106557e-02, -8.51367861e-02, -1.57826439e-01,  2.47479334e-01, 7.35986829e-02,  2.48634964e-01]],
                 [[-7.57155716e-02, -9.95359942e-02, -1.51629463e-01,  3.23129773e-01, -3.04444045e-01,  3.38968188e-01]],
                 [[-1.22465983e-01, -4.02057976e-01,  7.94960943e-04,  3.45858961e-01, -4.68735099e-01,  1.68319032e-01]]],
                [[[ 6.09288812e-02,  2.98400968e-01, -2.16035664e-01, -2.79128909e-01, -2.49070600e-02, -5.55095196e-01]],
                 [[ 2.45503932e-01,  6.62921220e-02, -2.05475941e-01, -2.87136555e-01, 3.86290103e-01,  5.74849062e-02]],
                 [[ 1.24202199e-01, -2.17195358e-02, -1.82127833e-01, -1.86002463e-01, 2.80802459e-01,  1.49210356e-02]],
                 [[-1.29296169e-01, -6.21485375e-02, -1.06248567e-02,  2.34563619e-01, 3.17129672e-01,  1.63708538e-01]],
                 [[-2.37110034e-01,  1.65250450e-01,  3.19648124e-02,  9.87904146e-02, 8.54154155e-02,  6.20286018e-02]]],
                [[[ 2.64266610e-01,  2.38384545e-01, -6.21965341e-02, -4.87131119e-01, 1.07345566e-01, -8.21851790e-01]],
                 [[ 2.18335569e-01,  2.59376556e-01,  1.54708410e-02, -1.79737866e-01, 1.16388410e-01, -5.31684220e-01]],
                 [[ 1.29895315e-01,  1.73845261e-01,  2.65149057e-01, -1.65988952e-01, 2.12555379e-01, -6.07524455e-01]],
                 [[-1.09471381e-01,  3.88127983e-01,  1.75194293e-01,  2.43370607e-01, 2.49128103e-01, -5.63060939e-01]],
                 [[-3.52162391e-01,  3.19904983e-01,  3.27137887e-01,  1.64880484e-01, 1.91338569e-01, -1.67967111e-01]]],
                [[[ 6.02521934e-02, -1.52366143e-02,  8.45037699e-02, -3.29230428e-01, -2.18086272e-01, -1.61821306e-01]],
                 [[ 2.27068260e-01,  1.35485604e-01,  2.89995044e-01, -1.97986871e-01, 1.32868424e-01, -3.01568419e-01]],
                 [[ 1.42230287e-01,  1.26459539e-01,  1.35385081e-01, -1.79014996e-01, 8.89067166e-03, -2.99682975e-01]],
                 [[ 1.11535482e-01,  7.12459981e-02,  1.85344905e-01,  1.14716344e-01, 6.82652965e-02, -1.63496628e-01]],
                 [[ 1.74192488e-02, -2.11645570e-02,  3.04915994e-01,  2.78766811e-01, 2.24847928e-01, -2.62693048e-01]]]]
Conv1_kernel = np.array(Conv1_kernel)
Pool1_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

'''
print(Conv1_kernel[0][0][0][0])
print(Conv1_kernel[0][1][0][0])
print(Conv1_kernel[0][2][0][0])
print(Conv1_kernel[1][0][0][0])
print(Conv1_kernel[1][1][0][0])
print(Conv1_kernel[2][0][0][0])
print(Conv1_kernel[2][1][0][0])
print(Conv1_kernel[2][2][0][0])
'''

Conv2_kernel = [[[[ 4.46148030e-03, -2.43060946e-01,  1.82859078e-01],            # (5,5,6,3)
                  [ 5.73413447e-02, -1.79924056e-01, -1.34274155e-01],
                  [-1.11200668e-01,  1.93545580e-01, -5.03118383e-03],
                  [ 1.94545552e-01,  1.04310684e-01,  5.04743196e-02],
                  [ 1.61035210e-01, -2.22076312e-01, -1.83698922e-01],
                  [ 1.09776214e-01, -6.30918369e-02, -1.09144427e-01]],
                 [[ 1.66212052e-01, -9.49484855e-02,  2.05616966e-01],
                  [ 1.19083099e-01,  2.60210992e-03,  3.32018249e-02],
                  [-3.72494161e-02,  9.93574411e-02,  4.38530818e-02],
                  [ 3.53315592e-01,  2.35935867e-01, -1.80826746e-02],
                  [ 1.75157294e-01,  7.85644278e-02, -2.57276833e-01],
                  [-1.04187012e-01, -1.14900813e-01, -1.69311598e-01]],
                 [[-3.60860233e-03,  7.67827183e-02,  3.21077257e-01],
                  [-4.30341735e-02, 2.28030458e-02,-1.24694742e-01],
                  [-2.73440689e-01, 1.60807729e-01,-1.80979595e-01],
                  [ 4.56692785e-01,  4.73841615e-02,  1.71454191e-01],
                  [ 2.81815767e-01,  1.43617451e-01, -1.96128637e-01],
                  [-1.90690622e-01, -2.81605422e-01, -4.41021137e-02]],
                 [[ 1.94376633e-01, -4.90035489e-02,  1.55250564e-01],
                  [-5.97173981e-02,  1.79853737e-02, -3.07055146e-01],
                  [-4.78641093e-02, -6.51851892e-02, -1.55271634e-01],
                  [ 2.67386168e-01, -3.70349400e-02,  1.19192161e-01],
                  [ 1.40169978e-01, -8.76651630e-02, -2.20318630e-01],
                  [-6.97066784e-01, -2.05059871e-02, -7.47444779e-02]],
                 [[ 1.65673018e-01,  3.31976041e-02, -9.33685824e-02],
                  [ 9.76644829e-03, -8.88296682e-03, -1.78523988e-01],
                  [ 1.09903105e-01, -2.97389686e-01, 1.63721328e-03],
                  [ 8.90138187e-03, -2.54530102e-01,  2.09646225e-02],
                  [ 7.52680078e-02,  9.90150794e-02, -1.28742591e-01],
                  [-3.59054297e-01,  2.79829443e-01, -1.00077577e-01]]],
                [[[-1.29625782e-01, -1.16864361e-01,  2.43602201e-01],
                  [ 1.52480423e-01,  7.45080933e-02, -1.17025144e-01],
                  [ 9.48696211e-02, -8.58374126e-03,  2.17507482e-02],
                  [-7.72828087e-02, -1.41286120e-01,  6.44770712e-02],
                  [ 1.51869044e-01, -8.63722712e-02, -2.28641585e-01],
                  [ 1.52700514e-01, -2.26914227e-01, -1.62951291e-01]],
                 [[-1.33052662e-01,  8.91523156e-03,  1.82868615e-01],
                  [ 2.25476727e-01,  1.91594884e-01, -4.13371287e-02],
                  [ 6.53660744e-02,  1.66390345e-01, -7.79683962e-02],
                  [ 6.98596658e-03, -9.50679556e-02, -1.01213962e-01],
                  [ 2.53964067e-01, -4.62014824e-02, -2.57423490e-01],
                  [ 7.45303780e-02, -2.23321736e-01,  1.22679219e-01]],
                 [[-1.95911713e-02,  1.70287654e-01,  3.81079875e-02],
                  [ 5.45673482e-02,  1.89345464e-01, -3.34087640e-01],
                  [-3.57262231e-02, -1.17123604e-01, -1.00682363e-01],
                  [ 1.59529313e-01,  5.27389646e-02,  1.70735374e-01],
                  [ 5.71238063e-02, -1.00585759e-01, -1.31461665e-01],
                  [-3.29621971e-01, -3.67517620e-02,  3.61060798e-01]],
                 [[ 1.86043113e-01,  7.06016943e-02, -1.37504622e-01],
                  [-1.15026623e-01, -8.08171406e-02, -2.01290429e-01],
                  [-1.80505171e-01, -1.00968689e-01, -2.57065922e-01],
                  [ 2.96369195e-01, -2.64048707e-02,  4.47448641e-02],
                  [ 2.83713996e-01,  1.68162256e-01, -1.12244435e-01],
                  [-4.45447177e-01,  1.39168054e-01, -1.52649119e-01]],
                 [[ 2.10947499e-01, -1.25490397e-01, -3.24189782e-01],
                  [-8.20837244e-02, -7.29732066e-02, -1.85146675e-01],
                  [-1.46288827e-01, -6.91832080e-02,  1.85829520e-01],
                  [ 1.93143144e-01, -1.15489855e-01,  6.80787265e-02],
                  [ 1.83288798e-01,  3.85825410e-02,  7.78506929e-03],
                  [-3.44356894e-01,  1.64550334e-01, -2.28331745e-01]]],
                [[[ 6.02760985e-02,  3.27797271e-02,  1.56385407e-01],
                  [ 1.52392656e-01, -4.88352701e-02, -4.10895161e-02],
                  [ 1.08795747e-01, -1.41858324e-01,  2.36514937e-02],
                  [-2.94398695e-01, -1.63445443e-01, -1.08941413e-01],
                  [ 1.83667898e-01, -8.91597420e-02, -1.26203120e-01],
                  [ 2.21730471e-02, -2.88801473e-02, -1.38151348e-01]],
                 [[ 7.43895322e-02, -9.31640640e-02,  1.92009464e-01],
                  [ 3.30580212e-02,  1.67621151e-01,  1.03326827e-01],
                  [ 1.45272121e-01, -1.20884500e-01,  8.66819918e-02],
                  [-3.03120553e-01, -1.68276295e-01,  1.42453536e-01],
                  [-1.03640564e-01,  5.45207076e-02,  2.38285647e-04],
                  [-2.91135192e-01,  7.13689476e-02,  2.20958859e-01]],
                 [[ 1.76585987e-01, -1.81971397e-02,  1.19804353e-01],
                  [-2.00220361e-01,  1.06862605e-01, -2.99832448e-02],
                  [ 1.07068472e-01, -1.09028660e-01, -1.42807901e-01],
                  [ 5.12155779e-02,  1.54463902e-01, -1.63206279e-01],
                  [-5.36084250e-02,  2.96080709e-01, -5.75776361e-02],
                  [-2.26433888e-01,  2.46767059e-01,  7.96443820e-02]],
                 [[-1.59939658e-02, -2.28865016e-02, -2.26515904e-01],
                  [-5.22559062e-02,  3.40456128e-01, -2.26670265e-01],
                  [-2.08191216e-01, -2.64145937e-02, -2.01001644e-01],
                  [ 2.46572062e-01,  6.09087991e-03,  1.86256301e-02],
                  [-4.72981185e-02,  2.47957766e-01, -1.65833145e-01],
                  [-2.73029149e-01,  3.71114820e-01, -9.85869020e-02]],
                 [[ 1.57899991e-01,  1.64293610e-02, -1.10490508e-01],
                  [-1.22783020e-01,  3.75140458e-01,  1.42274434e-02],
                  [-3.32721680e-01,  5.29803149e-02, -1.12240158e-01],
                  [ 4.58585352e-01, -1.65844932e-01,  1.91291213e-01],
                  [ 1.97771683e-01,  2.43736207e-01, -2.54018214e-02],
                  [-3.91964525e-01,  4.67040241e-01, -2.44016305e-01]]],
                [[[ 1.45044401e-01,  8.88259783e-02,  2.22703204e-01],
                  [ 1.89438015e-01,  1.14392020e-01,  1.66710421e-01],
                  [ 1.42902926e-01,  6.10344298e-02, -1.43264991e-03],
                  [-1.02467267e-02, -6.87396480e-03, -1.00042960e-02],
                  [-1.03194959e-01,  1.71727203e-02, -1.94432199e-04],
                  [ 5.30836917e-02,  1.77650079e-02, -1.96640998e-01]],
                 [[ 7.59854913e-02, -1.36784781e-02,  6.58905208e-02],
                  [-7.41148964e-02,  9.75294691e-03,  1.34717479e-01],
                  [-9.34582502e-02, -1.29915223e-01, -3.98084819e-02],
                  [ 2.28281524e-02, -2.40060300e-01,  7.63384849e-02],
                  [-2.51198024e-01,  7.19136968e-02,  3.39888260e-02],
                  [ 1.09661847e-01,  1.38019159e-01, -1.56892434e-01]],
                 [[-5.25103509e-02, -2.46554062e-01,  3.94752808e-02],
                  [-2.05721796e-01,  3.15680839e-02,  1.22758076e-01],
                  [-1.62131369e-01, -3.06537122e-01,  3.39966826e-02],
                  [ 1.74265549e-01, -7.05152797e-03, -5.80022931e-02],
                  [-6.99093193e-02,  2.52193272e-01,  4.99270968e-02],
                  [-2.06616253e-01,  4.84431446e-01, -2.95978367e-01]],
                 [[ 2.01963410e-01, -9.21719819e-02,  8.73498619e-04],
                  [ 7.58055151e-02,  5.26717678e-03, -8.90280306e-02],
                  [ 5.14510423e-02, -2.15486318e-01,  6.33772314e-02],
                  [ 2.34939083e-01,  1.96062446e-01, -8.18915144e-02],
                  [-1.28485039e-01, -1.83798391e-02,  2.48003393e-01],
                  [-3.66669804e-01,  4.63300794e-01, -2.89995372e-01]],
                 [[ 2.27360606e-01,  1.79621410e-02,  6.30379841e-02],
                  [ 1.35021821e-01,  9.80336815e-02,  1.02929279e-01],
                  [-1.02012768e-01, -5.26271202e-02,  2.12062821e-01],
                  [ 1.29118726e-01,  1.31497100e-01 , 1.80518001e-01],
                  [ 2.75496572e-01,  1.24823213e-01,  1.12401508e-01],
                  [-2.15923175e-01,  5.24682343e-01, -1.83883440e-02]]],
                [[[-2.49553677e-02, -2.41490945e-01,  5.74973412e-02],
                  [-8.41290727e-02,  2.35281765e-01,  3.16370465e-02],
                  [-1.04939409e-01, -1.52145937e-01,  1.23863772e-01],
                  [ 6.01596832e-02,  1.82747915e-01, -6.35440424e-02],
                  [-1.54443666e-01,  2.97095776e-01,  7.88350478e-02],
                  [ 3.32017303e-01,  1.34418994e-01, -9.99353826e-03]],
                 [[-1.87003255e-01, -2.63412982e-01,  9.68348235e-02],
                  [ 1.84518192e-02,  2.79142261e-01,  1.84531540e-01],
                  [ 2.23084330e-03,  2.61985548e-02,  2.03359574e-01],
                  [-1.70044508e-02, -9.67519134e-02, -7.77086392e-02],
                  [-3.99361700e-02,  3.04613382e-01,  2.61705875e-01],
                  [ 8.29325765e-02,  3.33145469e-01,  5.12788594e-02]],
                 [[ 4.83596092e-03, -6.12523705e-02,  2.36663036e-02],
                  [-1.47255406e-01,  2.55434662e-02,  1.35068417e-01],
                  [-2.36879632e-01,  8.61347653e-03, -7.36168073e-03],
                  [ 1.34673655e-01, -3.11976969e-02,  2.77323071e-02],
                  [-3.68976370e-02,  2.38494650e-01,  3.52538556e-01],
                  [-1.06586613e-01,  4.50379074e-01,  1.36053771e-01]],
                 [[ 1.55939251e-01, -1.34711191e-01, -1.01322848e-02],
                  [-2.78911889e-02,  1.64967149e-01,  2.18779057e-01],
                  [-1.29998531e-02, -1.29886180e-01,  2.92970896e-01],
                  [-3.63288224e-02,  1.46652618e-02,  1.16084881e-01],
                  [-7.92094618e-02, -1.30063429e-01,  3.39011014e-01],
                  [-2.15137824e-01,  4.46359932e-01,  3.55196655e-01]],
                 [[ 1.12604082e-01, -1.34613067e-01, -5.26728593e-02],
                  [-6.50925329e-04,  1.34872524e-02,  2.66594529e-01],
                  [ 1.08478405e-01, -1.06381148e-01,  4.47653562e-01],
                  [ 1.24460444e-01, -6.48147911e-02,  7.65434355e-02],
                  [ 2.63863616e-02, -2.06157222e-01, -2.69554730e-04],
                  [ 2.90843427e-01,  1.78602770e-01,  7.68283382e-02]]]]
Conv2_kernel = np.array(Conv2_kernel)
Pool2_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

Conv1 = Convolution(reshape_test_data,Conv1_kernel)
Max_pooling1 = Max_pooling(Conv1,Pool1_kernel_filter_size)

Conv2 = Convolution(Max_pooling1,Conv2_kernel)
Max_pooling2= Max_pooling(Conv2,Pool2_kernel_filter_size)
Fully_connected = Fully_connected(Max_pooling2)

''' # 4행 5열(4,5)일때 출력한 값(필터개수 6개)
[[[[ 0.18785302  0.2737564   0.02352315 -0.22898602 -0.03165494 -0.10813242]]
[[-0.21621639  0.09240021  0.17102224 -0.3154531  -0.24477777 -0.21968114]]
[[-0.047902   -0.05450074 -0.11579917 -0.2578172  -0.18218452 0.10793413]]
[[ 0.02836869  0.04933267  0.26139185  0.02720428 -0.2671 0.05570612]]
[[-0.08774811 -0.4403676  -0.04567708  0.34815112 -0.27944946 0.22130887]]]
[[[ 0.31017354  0.03930232 -0.05199112 -0.62546986 -0.02265492 -0.17029339]]
[[-0.17281215  0.22783281 -0.03061478 -0.37058046 -0.18641353 -0.00832481]]
[[-0.46741325  0.10780349  0.38679814  0.00630076 -0.36132997 0.2338634 ]]
[[-0.42613938  0.15558927  0.09021978  0.18976699 -0.13050072 0.35606673]]
[[-0.28344986  0.07521104 -0.04253848  0.42007014 -0.2699382 0.2553394 ]]]
[[[ 0.10245772  0.34458637  0.00501653 -0.23863828 -0.06058068 0.2754111 ]]
[[ 0.03024862  0.19908659  0.3359361  -0.28148893  0.09492078 0.39281678]]
[[-0.19558364  0.1293984   0.26862302 -0.12896696  0.17023687 0.23006867]]
[[ 0.24517381  0.41833332  0.30300978  0.4214005   0.12438442 0.05506228]]
[[ 0.14820985  0.35075486 -0.4113046   0.10445357  0.30912343 0.20575558]]]
[[[ 0.21927138  0.07816353  0.40818405 -0.06957868  0.25261834 0.2756156 ]]
[[ 0.24993907 -0.10606384 -0.02329633 -0.10104813  0.36926147  0.14633407]]
[[ 0.32394257 -0.266679    0.11738923 -0.00370438  0.30578223  0.23860924]]
[[ 0.10616782  0.19271463  0.16182242  0.39618737  0.3658185 0.10984062]]
[[ 0.11061162  0.3487492   0.04390993  0.36420214  0.2449066 0.10274311]]]]
'''