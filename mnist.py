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
'''


'''
import numpy as np

def relu(x):
    return np.maximum(0,x)

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

    #print(input[1:6, 8:13, 0])

    conv_sum = []
    if input_node == 1:
        for i in range(num_filter):
            for j in range(input_node):
                for k in range(num_input):
                    conv = []
                    for l in range(new_width):
                        for m in range(new_height):
                            conv.append((input[l:l + kernel_width, m:m + kernel_height, k] * new_kernel[i][j]).sum())
                conv_sum.append(conv)

        conv_sum = np.array(conv_sum).reshape(new_width, new_height, num_filter, num_channel)
        conv_sum = relu(conv_sum)
        print(conv_sum.shape)
        #print(conv_sum)
        return conv_sum

    elif input_node > 1:
        for i in range(num_filter):
            for j in range(input_node):
                for k in range(num_input):
                    conv = []
                    for l in range(new_width):
                        for m in range(new_height):
                            conv.append((input[l:l + kernel_width, m:m + kernel_height, k] * new_kernel[i][j]).sum())
                conv_sum.append(conv)

        conv_sum = np.array(conv_sum).reshape(num_filter,input_node,new_width * new_height)

        filter_sum = 0
        all_filter_sum = []
        for i in range(num_filter):
            for j in range(new_width * new_height):
                for k in range(input_node):
                    filter_sum += conv_sum[i][k][j]
                all_filter_sum.append(filter_sum)
                filter_sum = 0
        all_filter_sum = np.array(all_filter_sum)


        all_filter_sum = np.array(all_filter_sum).reshape(new_width, new_height, num_filter, num_channel)
        all_filter_sum = relu(all_filter_sum)
        print(all_filter_sum.shape)
        return all_filter_sum

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
    pool = np.array(pool).reshape(new_width, new_height, num_input, num_channel)
    print(pool.shape)
    return pool

def Fully_connected(input):
    input_width, input_height, num_input, num_channel = input.shape
    flatten = input.reshape(num_channel * num_input * input_width * input_height)
    print(flatten.shape)
    #print(flatten)

test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
test_data = test_data / 255
reshape_test_data = test_data[1:].reshape(28,28,1,1)
np.set_printoptions(threshold=np.inf) # 모든 배열의 수 출력

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
'''



'''
import numpy as np

def relu(x):
    return np.maximum(0,x)

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

    print(input[1:6, 8:13, 0])
    #print(new_kernel[0][0])

    conv_sum = []
    if input_node == 1:
        for i in range(num_filter):
            for j in range(input_node):
                for k in range(num_input):
                    conv = []
                    for l in range(new_height):
                        for m in range(new_width):
                            conv.append((input[l:l + kernel_height, m:m + kernel_width, k] * new_kernel[i][j]).sum())
                conv_sum.append(conv)

        conv_sum = np.array(conv_sum).reshape(new_width, new_height, num_filter, num_channel)
        conv_sum = relu(conv_sum)
        print(conv_sum.shape)
        #print(conv_sum)
        return conv_sum

    elif input_node > 1:
        for i in range(num_filter):
            for j in range(input_node):
                for k in range(num_input):
                    conv = []
                    for l in range(new_height):
                        for m in range(new_width):
                            conv.append((input[l:l + kernel_height, m:m + kernel_width, k] * new_kernel[i][j]).sum())
                conv_sum.append(conv)

        conv_sum = np.array(conv_sum).reshape(num_filter,input_node,new_width * new_height)

        filter_sum = 0
        all_filter_sum = []
        for i in range(num_filter):
            for j in range(new_width * new_height):
                for k in range(input_node):
                    filter_sum += conv_sum[i][k][j]
                all_filter_sum.append(filter_sum)
                filter_sum = 0

        all_filter_sum = np.array(all_filter_sum).reshape(new_width, new_height, num_filter, num_channel)
        all_filter_sum = relu(all_filter_sum)
        print(all_filter_sum.shape)
        return all_filter_sum

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
    pool = np.array(pool).reshape(new_width, new_height, num_input, num_channel)
    print(pool.shape)
    return pool

def Fully_connected(input):
    input_width, input_height, num_input, num_channel = input.shape
    flatten = input.reshape(num_channel * num_input * input_width * input_height)
    print(flatten.shape)
    #print(flatten)

test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
test_data = test_data / 255
reshape_test_data = test_data[1:].reshape(28,28,1,1)
np.set_printoptions(threshold=np.inf) # 모든 배열의 수 출력

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
'''



'''
import numpy as np

def relu(x):
    return np.maximum(0,x)

def Convolution(input,kernel):
    num_channel, num_input, input_width, input_height = input.shape
    num_filter, input_node, kernel_width, kernel_height = kernel.shape
    new_width = input_width - kernel_width + 1
    new_height = input_height - kernel_height + 1
    new_kernel = []
    for i in range(num_filter):
        for j in range(input_node):
            for k in range(kernel_width):
                for l in range(kernel_height):
                    new_kernel.append(kernel[i][j][k][l])
    new_kernel = np.array(new_kernel).reshape(num_filter, input_node, kernel_height, kernel_width)

    #print(input[1:6, 8:13, 0])
    #print(new_kernel[0][0])

    conv_sum = []
    if input_node == 1:
        for i in range(num_filter):
            for j in range(input_node):
                for k in range(num_input):
                    conv = []
                    for l in range(new_height):
                        for m in range(new_width):
                            conv.append((input[0, k, l:l + kernel_height, m:m + kernel_width] * new_kernel[i][j]).sum())
                conv_sum.append(conv)
        conv_sum = np.array(conv_sum)

        conv_sum = np.array(conv_sum).reshape(num_channel, num_filter, new_width, new_height)
        conv_sum = relu(conv_sum)
        print(conv_sum.shape)
        print(conv_sum)
        return conv_sum

    elif input_node > 1:
        for i in range(num_filter):
            for j in range(input_node):
                for k in range(num_input):
                    conv = []
                    for l in range(new_height):
                        for m in range(new_width):
                            conv.append((input[0, k, l:l + kernel_height, m:m + kernel_width] * new_kernel[i][j]).sum())
                conv_sum.append(conv)

        conv_sum = np.array(conv_sum).reshape(num_filter,input_node,new_width * new_height)

        filter_sum = 0
        all_filter_sum = []
        for i in range(num_filter):
            for j in range(new_width * new_height):
                for k in range(input_node):
                    filter_sum += conv_sum[i][k][j]
                all_filter_sum.append(filter_sum)
                filter_sum = 0

        all_filter_sum = np.array(all_filter_sum).reshape(num_channel, num_filter, new_width, new_height)
        all_filter_sum = relu(all_filter_sum)
        print(all_filter_sum.shape)
        return all_filter_sum

def Max_pooling(input,kernel_size):
    num_channel, num_input, input_width, input_height = input.shape
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
    print(pool.shape)
    return pool

def Fully_connected(input):
    input_width, input_height, num_input, num_channel = input.shape
    flatten = input.reshape(num_channel * num_input * input_width * input_height)
    print(flatten.shape)
    #print(flatten)

test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
test_data = test_data / 255
reshape_test_data = test_data[1:].reshape(1,1,28,28)
np.set_printoptions(threshold=np.inf) # 모든 배열의 수 출력

Conv1_kernel = [[[[-0.0680,  0.0275, -0.2677, -0.2375,  0.2525],   # (6,1,5,5)
          [-0.1672, -0.2823, -0.2264, -0.0311,  0.2349],
          [-0.0725,  0.0390,  0.1260,  0.2669,  0.1895],
          [ 0.2636, -0.0127,  0.2255,  0.0317,  0.3159],
          [ 0.1191, -0.0783, -0.0165,  0.3605,  0.3640]]],

        [[[ 0.3472,  0.1254,  0.1107,  0.2817,  0.2512],
          [ 0.3364,  0.4145,  0.3100,  0.2027,  0.4227],
          [ 0.1939,  0.2157,  0.2623,  0.1714,  0.1232],
          [ 0.3450,  0.0264,  0.0801, -0.1014,  0.0415],
          [ 0.1158, -0.0388, -0.3904, -0.3124, -0.3086]]],

        [[[ 0.0525, -0.0718,  0.3059,  0.0913, -0.0510],
          [-0.1374,  0.2809,  0.0937, -0.0231, -0.1671],
          [-0.1174,  0.2682,  0.3095,  0.1708,  0.0542],
          [-0.0656,  0.0585, -0.0375,  0.3021,  0.0639],
          [-0.1630, -0.0181,  0.0806,  0.2859,  0.3343]]],

        [[[ 0.0998,  0.2991,  0.2953,  0.3074,  0.0406],
          [ 0.1017,  0.2535,  0.1534, -0.0839, -0.2580],
          [ 0.0445,  0.4135,  0.1037, -0.1744, -0.0603],
          [ 0.1078,  0.1993,  0.0512,  0.0627, -0.2375],
          [ 0.3078,  0.1889,  0.0572,  0.1751,  0.2362]]],

        [[[ 0.0967,  0.0459,  0.1245,  0.1683,  0.1238],
          [-0.0481,  0.0496,  0.1173,  0.2847,  0.3487],
          [ 0.2876,  0.0921,  0.2034,  0.2469,  0.1842],
          [ 0.0704,  0.1705,  0.3190,  0.3281,  0.1892],
          [-0.0976,  0.0431,  0.2050,  0.2965,  0.0706]]],

        [[[ 0.0121,  0.0611,  0.3250,  0.3247,  0.0844],
          [-0.1208,  0.1395,  0.1723,  0.2998,  0.2486],
          [ 0.0996,  0.2588,  0.0566,  0.3004,  0.2866],
          [ 0.1867,  0.1692,  0.3370,  0.2410, -0.0976],
          [ 0.2576,  0.0457,  0.1106,  0.0760,  0.1850]]]]
Conv1_kernel = np.array(Conv1_kernel)
Pool1_kernel_filter_size = (2,2)                             # 2*2 down sampling


Conv2_kernel = [[[[-4.8119e-02,  1.5286e-01,  6.0189e-02, -1.1039e-01, -1.2659e-01],     # (3,6,5,5)
          [ 6.3107e-02,  2.1599e-01, -3.7750e-02, -2.9188e-01, -1.1176e-01],
          [ 2.3574e-01,  1.6669e-01, -2.7364e-01, -1.9998e-01,  1.0275e-01],
          [ 2.6009e-01,  2.9236e-02, -1.1253e-01, -2.1788e-02, -3.8051e-03],
          [ 1.1041e-01,  2.4858e-02, -1.2534e-01,  2.8029e-02,  2.3566e-02]],

         [[-2.7559e-02, -1.7111e-01, -6.1218e-02,  1.4126e-01,  1.5385e-01],
          [-2.8809e-01, -1.4490e-01,  6.2571e-02,  8.1776e-02, -4.6613e-02],
          [-3.1152e-01, -1.3147e-02,  6.1851e-02,  5.5753e-02, -2.0084e-01],
          [-9.4645e-02,  1.0374e-02,  8.9500e-02, -1.0302e-01, -9.3832e-02],
          [-1.1037e-01,  3.2230e-02,  1.1753e-01, -3.8202e-02, -1.2300e-02]],

         [[-9.0114e-02,  1.0128e-01,  4.7368e-02, -6.5659e-02, -6.1816e-02],
          [-4.9881e-02,  1.2651e-01,  3.4634e-03,  2.7452e-02, -1.2404e-01],
          [ 3.7933e-02,  3.0420e-02,  7.1624e-02, -1.4592e-01, -1.5152e-01],
          [-7.9277e-02,  6.0234e-02,  1.2562e-01, -1.2024e-01,  7.8983e-03],
          [ 4.7702e-03,  2.4638e-02,  6.9159e-02,  4.1420e-02,  8.3481e-02]],

         [[-7.5516e-02, -1.1002e-01,  4.8978e-02,  1.2439e-01,  7.5344e-02],
          [-2.0632e-01, -1.2288e-01,  1.3696e-01,  1.2045e-01, -5.2961e-03],
          [-2.8662e-01, -8.3719e-02,  2.1172e-01,  1.1384e-01, -1.8208e-01],
          [-1.5500e-01,  1.3252e-01,  1.4349e-01, -8.3343e-02, -6.0864e-02],
          [-1.1610e-01,  1.4905e-01,  1.6403e-02, -1.5611e-02, -3.1249e-03]],

         [[-3.2132e-02,  8.9801e-02,  6.3060e-02, -3.3810e-02, -6.5642e-02],
          [-1.0903e-01,  1.8437e-01,  7.5385e-02, -3.0576e-02, -1.4576e-01],
          [-2.1844e-02,  2.1331e-01, -2.9893e-02, -1.5205e-01, -5.5007e-02],
          [ 1.6955e-01,  9.6739e-02, -5.0497e-02, -1.2256e-01, -3.6003e-02],
          [ 7.0790e-02,  6.7593e-02, -8.6049e-02, -5.5031e-02, -5.4564e-03]],

         [[-1.4586e-01, -1.1218e-02,  1.6932e-01,  1.0219e-01,  3.1556e-02],
          [-1.2864e-01,  1.4716e-01,  1.7311e-01, -8.5117e-03, -7.0402e-02],
          [ 1.1241e-02,  1.6088e-01,  3.5008e-02, -2.7245e-02, -2.1770e-01],
          [ 6.4591e-02,  1.3779e-01, -3.5553e-02, -8.6416e-02, -8.6409e-02],
          [ 4.0320e-02,  1.3303e-01, -1.1103e-01, -1.3389e-02, -1.0507e-01]]],


        [[[ 5.6951e-02,  9.2089e-02,  4.6883e-02,  1.0300e-01, -3.5433e-02],
          [-5.8432e-03,  3.8902e-02,  6.1181e-02,  1.6377e-01, -4.6364e-02],
          [-2.0150e-01, -3.3938e-02,  1.0483e-01, -4.2678e-02, -9.0905e-02],
          [-8.5068e-02, -5.2119e-03,  6.6591e-02,  2.6757e-02, -4.2918e-02],
          [ 3.4387e-02,  1.3075e-02, -3.5953e-02, -8.3483e-02,  6.0684e-02]],

         [[-7.0556e-02, -8.2841e-04,  1.2328e-01,  9.9668e-02,  1.3220e-01],
          [ 9.8942e-02,  7.4259e-02,  5.0377e-02, -9.0004e-02,  7.6703e-02],
          [ 1.2049e-01, -5.6804e-02, -5.6773e-02,  4.0932e-02,  1.3972e-01],
          [-8.4423e-02, -1.8802e-01, -7.3185e-02,  1.2429e-01,  4.5991e-02],
          [-7.2963e-02, -5.4057e-02,  6.2816e-02, -1.8397e-02,  2.6666e-02]],

         [[ 1.1319e-01,  1.3369e-01,  8.2279e-02,  2.1065e-02,  1.3108e-01],
          [ 4.2289e-02, -2.5033e-02,  1.0796e-01,  1.5562e-01,  1.6654e-02],
          [-1.1149e-01,  2.7132e-02,  1.2975e-01,  1.2873e-01,  2.3393e-02],
          [-2.1930e-02,  1.5405e-02, -1.1819e-02,  2.2001e-01,  1.9701e-02],
          [-4.9434e-02, -2.7365e-02, -1.5226e-03,  1.8076e-01,  1.1241e-01]],

         [[-3.4150e-02, -5.6084e-02,  1.3239e-01,  5.7209e-02,  1.4454e-01],
          [-5.9983e-02,  8.1980e-02,  6.7557e-02, -2.8821e-02,  2.1180e-02],
          [ 5.2551e-02,  1.0824e-02, -1.1012e-01,  8.3728e-02,  1.0333e-01],
          [ 2.4993e-03, -2.0828e-03, -1.5606e-01,  1.4106e-01,  1.6446e-01],
          [-1.0245e-02,  6.1105e-03, -1.1904e-02, -1.2074e-02,  3.0205e-02]],

         [[ 7.3539e-02,  1.0966e-01,  1.4968e-01,  2.3607e-02,  1.1394e-01],
          [-4.4291e-02,  3.0694e-02,  1.1250e-01,  7.4067e-02,  2.2008e-03],
          [-1.2823e-01, -6.8196e-02,  1.2378e-01,  1.5545e-01, -1.8075e-02],
          [-1.2631e-01, -5.0417e-02,  6.1530e-02,  1.1671e-01,  7.6057e-02],
          [-1.3560e-01,  4.9783e-02,  1.0163e-01, -3.2624e-02,  9.0872e-02]],

         [[ 3.1963e-02,  8.8982e-02,  1.6339e-01,  1.4938e-01,  9.3292e-02],
          [ 5.7015e-02,  1.1725e-02,  3.0390e-03,  1.1023e-01,  7.5770e-02],
          [ 7.7394e-03, -2.3976e-02,  1.2182e-02,  1.2343e-01,  1.0455e-01],
          [-1.7241e-01, -1.3388e-01,  1.1147e-01,  1.3886e-01, -2.3252e-02],
          [-3.4884e-02, -4.0318e-02,  8.3365e-02,  1.2608e-01,  2.1375e-05]]],


        [[[ 1.6221e-01,  1.5761e-01,  1.2696e-01,  3.0173e-02, -5.8833e-02],
          [ 1.6438e-01,  1.4044e-01,  7.1904e-02, -3.7639e-03,  1.0204e-01],
          [-1.4652e-01, -1.0632e-01, -1.4384e-01, -1.0517e-01, -1.4022e-02],
          [-2.1268e-01, -1.1189e-01, -1.6890e-01, -3.5848e-02, -5.1476e-02],
          [-2.7879e-02,  5.3855e-02,  1.3351e-01,  1.3948e-01,  1.3332e-02]],

         [[-6.4880e-02, -1.0405e-01,  6.7903e-02,  1.6255e-01,  9.9083e-02],
          [ 4.2538e-02,  9.4716e-02,  1.2211e-01,  1.6215e-01,  9.6690e-02],
          [ 2.2809e-01,  1.6303e-01,  2.0093e-01,  2.2034e-01,  1.8770e-01],
          [-3.4723e-02, -4.3723e-02, -2.7061e-02, -2.9601e-02, -1.0950e-01],
          [-2.4164e-01, -2.5869e-01, -2.3255e-01, -1.7038e-01, -1.7056e-01]],

         [[ 1.5338e-01,  9.6567e-02,  1.2509e-01,  1.1608e-01,  1.5900e-02],
          [ 1.0472e-01,  1.0088e-01,  6.4225e-02,  7.9112e-02,  1.1326e-01],
          [-2.9044e-02, -1.0142e-01, -1.0543e-01, -1.2770e-01,  6.7645e-02],
          [-2.5260e-01, -2.0164e-01, -1.8367e-01, -1.1848e-01, -4.8070e-02],
          [-1.9225e-01, -5.6690e-02, -1.6740e-01, -7.8585e-02,  2.7054e-02]],

         [[ 6.1302e-02,  5.3365e-02, -1.1081e-02,  1.5511e-01,  1.5110e-01],
          [-8.7929e-02, -5.5436e-02,  1.4397e-01,  1.5590e-01,  1.3631e-01],
          [-1.8824e-01,  3.2539e-03,  6.2167e-02, -7.1944e-03,  5.3520e-02],
          [ 1.3477e-01,  1.3016e-01, -3.0380e-02, -5.5726e-02, -8.4263e-02],
          [-4.7335e-02, -8.7781e-02, -8.3166e-02, -1.0813e-01, -7.4285e-02]],

         [[ 8.6422e-02,  3.4313e-02,  1.4375e-01,  1.9636e-01,  8.1815e-02],
          [ 1.0095e-01,  1.7901e-01,  1.6213e-01,  1.9865e-01,  1.0545e-01],
          [ 2.8260e-02,  1.0750e-01,  4.9622e-02,  1.0478e-01,  9.7823e-02],
          [-1.2988e-01, -1.0404e-01, -1.1884e-01, -8.7504e-02, -8.6685e-02],
          [-8.6361e-02, -1.6208e-01, -1.4766e-01, -1.3417e-01, -6.1367e-02]],

         [[-9.6943e-02, -4.1657e-02,  1.1709e-01,  6.8367e-02,  1.7747e-01],
          [ 3.0086e-02,  1.1094e-01,  6.7883e-02,  1.9036e-01,  1.2419e-01],
          [ 1.2825e-02,  1.5314e-01,  1.6634e-01,  1.5974e-02, -3.0327e-02],
          [-5.9907e-02, -1.4156e-01, -1.9872e-01, -1.8649e-01, -1.4756e-01],
          [-8.2826e-02, -1.3948e-01, -1.8027e-01, -3.6667e-02, -1.6836e-02]]]]
Conv2_kernel = np.array(Conv2_kernel)
Pool2_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

Conv1 = Convolution(reshape_test_data,Conv1_kernel)
Max_pooling1 = Max_pooling(Conv1,Pool1_kernel_filter_size)

Conv2 = Convolution(Max_pooling1,Conv2_kernel)
Max_pooling2= Max_pooling(Conv2,Pool2_kernel_filter_size)
Fully_connected = Fully_connected(Max_pooling2)
'''


import numpy as np

def relu(x):
    return np.maximum(0,x)

def Convolution(input,kernel, bias):
    num_channel, num_input, input_width, input_height = input.shape
    num_filter, input_node, kernel_width, kernel_height = kernel.shape
    new_width = input_width - kernel_width + 1
    new_height = input_height - kernel_height + 1
    new_kernel = []
    for i in range(num_filter):
        for j in range(input_node):
            for k in range(kernel_width):
                for l in range(kernel_height):
                    new_kernel.append(kernel[i][j][k][l])
    new_kernel = np.array(new_kernel).reshape(num_filter, input_node, kernel_width, kernel_height)

    conv_sum = []
    if input_node == 1:
        for i in range(num_filter):
            for j in range(input_node):
                conv = []
                for k in range(new_height):
                    for l in range(new_width):
                        conv.append((input[0, j, k:k + kernel_height, l:l + kernel_width] * new_kernel[i][j]).sum())
                conv_sum.append(conv)

        conv_sum = np.array(conv_sum).reshape(num_channel, num_filter, new_width, new_height)

        #bias
        for i in range(num_channel):
            for j in range(num_filter):
                for k in range(new_width):
                    for l in range(new_height):
                        conv_sum[i][j][k][l] += bias[j]

        #conv_sum = relu(conv_sum)
        print('-------------------------------conv1 layer-------------------------------------')
        print(conv_sum.shape)
        print(conv_sum)
        return conv_sum

    elif input_node > 1:
        for i in range(num_filter):
            for j in range(input_node):
                conv = []
                for k in range(new_height):
                    for l in range(new_width):
                        conv.append((input[0, j, k:k + kernel_height, l:l + kernel_width] * new_kernel[i][j]).sum())
                conv_sum.append(conv)

        conv_sum = np.array(conv_sum).reshape(num_filter,input_node,new_width * new_height)
        filter_sum = 0
        all_filter_sum = []
        for i in range(num_filter):
            for j in range(new_width * new_height):
                for k in range(input_node):
                    filter_sum += conv_sum[i][k][j]
                all_filter_sum.append(filter_sum)
                filter_sum = 0

        all_filter_sum = np.array(all_filter_sum).reshape(num_channel, num_filter, new_width, new_height)

        #bias
        for i in range(num_channel):
            for j in range(num_filter):
                for k in range(new_width):
                    for l in range(new_height):
                        all_filter_sum[i][j][k][l] += bias[j]

        #all_filter_sum = relu(all_filter_sum)
        print('---------------------------------conv2 layer------------------------------------')
        print(all_filter_sum.shape)
        print(all_filter_sum)
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
    print(pool)

    return pool

def Fully_connected(input):
    num_channel, num_input, input_width, input_height = input.shape
    flatten = input.reshape(num_channel * num_input * input_width * input_height)
    print('-------------------------fully connected layer----------------------------------')
    print(flatten.shape)
    #print(flatten)

test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
test_data = test_data / 255
reshape_test_data = test_data[1:].reshape(1,1,28,28)
np.set_printoptions(threshold=np.inf) # 모든 배열의 수 출력

# (6,1,5,5)
Conv1_kernel = [[[[-0.1135,  0.1017, -0.0085, -0.1047,  0.1724],
          [-0.1487, -0.0879, -0.1349, -0.1859, -0.0333],
          [-0.0224, -0.0053, -0.1196, -0.1286,  0.1696],
          [ 0.0039, -0.0033, -0.0034, -0.0715,  0.0073],
          [ 0.1119,  0.0447, -0.1326,  0.1059,  0.1480]]],

        [[[-0.1368,  0.0296,  0.0976, -0.0723, -0.1766],
          [ 0.0240,  0.1553,  0.1562,  0.0591,  0.0368],
          [ 0.1308, -0.0819, -0.1609, -0.0954,  0.1303],
          [ 0.1383,  0.1440, -0.0941, -0.0368, -0.0034],
          [-0.1098, -0.0900,  0.1325, -0.0803, -0.1644]]],

        [[[-0.0146, -0.1663,  0.1801, -0.1782,  0.1788],
          [ 0.1787, -0.1623,  0.1815, -0.0624,  0.1891],
          [-0.0869,  0.0367,  0.0659, -0.0272,  0.1210],
          [-0.1887,  0.1407, -0.1079,  0.1740, -0.0791],
          [ 0.0338,  0.1147, -0.0985,  0.0605, -0.1682]]],

        [[[ 0.0656, -0.1988,  0.1400, -0.0320,  0.0483],
          [ 0.0916,  0.1834,  0.1958,  0.1365,  0.0330],
          [ 0.1107, -0.0340,  0.0575,  0.0474,  0.1858],
          [ 0.0127,  0.0819,  0.1873,  0.0214, -0.1771],
          [ 0.0509,  0.1305, -0.1391, -0.1766, -0.1309]]],

        [[[ 0.0306, -0.1070, -0.0110, -0.1921, -0.0642],
          [ 0.0641, -0.0055,  0.0171, -0.0269, -0.0273],
          [-0.1531, -0.0859,  0.1836,  0.0293,  0.1305],
          [-0.1869,  0.1018,  0.0859, -0.0901, -0.0454],
          [-0.0145, -0.1158,  0.1588,  0.0772,  0.0759]]],

        [[[-0.1947,  0.0731,  0.0990, -0.1401, -0.0557],
          [ 0.1664, -0.0430,  0.0970,  0.1520, -0.1057],
          [-0.1821,  0.0601, -0.0387,  0.0819, -0.0970],
          [-0.1516,  0.0349,  0.0846, -0.0148,  0.0499],
          [ 0.0246,  0.1539,  0.0055, -0.0205, -0.1515]]]]
Conv1_kernel_bias = [-0.0136,  0.0704, -0.0228, -0.0992, -0.1615, -0.0885]
Conv1_kernel = np.array(Conv1_kernel)
Pool1_kernel_filter_size = (2,2)    # 2*2 down sampling


# (3,6,5,5)
Conv2_kernel = [[[[ 5.8559e-02,  3.9410e-02, -4.7156e-02, -8.3816e-03, -7.1896e-02],
          [-6.1733e-02,  1.2220e-02,  4.3487e-02, -7.5139e-02, -3.8978e-03],
          [ 7.7162e-02, -3.4915e-02, -2.0572e-03,  7.7454e-02, -5.4742e-02],
          [ 3.9822e-02, -5.7787e-04,  3.0505e-02, -1.3722e-02,  2.5333e-02],
          [-2.5473e-02,  7.2699e-02, -4.2849e-02,  2.2749e-02,  6.5750e-02]],

         [[-3.4368e-02, -2.8858e-02,  6.2048e-04,  8.3040e-03, -6.0023e-02],
          [ 3.0821e-02,  1.3655e-02, -5.0018e-02, -7.1342e-02,  4.2839e-02],
          [ 4.8074e-02,  5.5580e-02, -8.0828e-02, -4.2794e-02, -5.3081e-02],
          [-3.9789e-02, -6.0365e-03, -1.1223e-02,  7.0002e-02, -3.5552e-02],
          [ 4.2212e-03,  4.9356e-02, -5.3851e-02, -1.7297e-03,  6.5000e-02]],

         [[-1.0117e-02, -3.0434e-02, -6.7490e-02, -5.7754e-02, -6.4642e-02],
          [-1.0656e-02, -6.9722e-02,  7.8094e-02, -6.3639e-02,  2.0752e-02],
          [-3.1583e-02, -3.5253e-03,  6.2080e-02, -1.4306e-03,  4.2495e-02],
          [ 4.7410e-02,  5.5965e-02,  4.8278e-02,  6.9646e-03,  3.1034e-02],
          [-2.7092e-02, -3.1495e-02, -2.8440e-03,  2.2520e-02,  3.3046e-02]],

         [[ 3.2613e-02, -1.9486e-02,  3.7424e-02, -4.0298e-02, -1.9903e-02],
          [ 1.0093e-02,  2.2823e-02,  2.4957e-02, -2.2215e-02, -5.2616e-02],
          [-1.7867e-03, -4.2815e-02,  7.2249e-02,  1.5564e-02,  6.1841e-02],
          [-1.4910e-02,  2.9381e-03, -2.1695e-02,  6.6981e-02,  2.8561e-02],
          [-7.3251e-02,  2.9423e-02, -2.5367e-02,  2.1645e-02, -3.1620e-04]],

         [[ 3.1046e-02, -8.0975e-02,  2.2352e-02, -1.4511e-02,  8.1936e-03],
          [-7.9386e-02, -4.9647e-02, -8.8276e-03, -5.8831e-02, -7.1788e-03],
          [ 6.2264e-02,  4.3093e-02,  3.8869e-02,  6.9617e-03, -7.4981e-02],
          [-5.3709e-02, -7.5012e-02, -1.2158e-02,  7.4664e-02,  6.8876e-02],
          [ 3.3478e-02,  7.9263e-03,  4.7615e-03,  3.1143e-02,  1.5193e-02]],

         [[ 2.3153e-03, -5.4402e-03,  2.1585e-02,  7.8713e-03,  7.9514e-02],
          [-1.4483e-02,  3.3438e-02, -5.0100e-02, -3.5543e-02,  4.4559e-02],
          [ 1.0036e-02, -2.6423e-02,  4.2192e-02,  8.0544e-02, -7.9997e-02],
          [ 8.7107e-03,  6.9471e-02,  3.1722e-02, -1.8181e-02, -1.2901e-02],
          [-4.3207e-02,  7.5776e-02, -6.5725e-02, -5.9717e-02, -2.5644e-02]]],


        [[[ 4.0829e-03, -7.3924e-02,  6.0598e-02,  3.9426e-02, -3.7959e-02],
          [-2.7576e-02,  5.2575e-02,  4.1949e-02,  7.1558e-02,  2.9180e-02],
          [ 5.4066e-02, -1.2576e-02, -8.0155e-02,  1.6440e-03, -5.6861e-03],
          [-6.3395e-02, -3.7515e-02, -7.5864e-03, -5.4104e-02,  2.7501e-02],
          [ 4.7540e-02,  2.2728e-02,  7.8770e-02,  2.8879e-02,  4.2570e-02]],

         [[-1.8966e-02, -4.7999e-02, -8.0852e-02, -7.4347e-02,  7.7408e-02],
          [-6.0232e-02, -2.3681e-02,  6.7777e-03,  6.3487e-02, -6.6229e-02],
          [-2.0598e-02, -7.5999e-02, -4.0173e-02, -1.6784e-02,  5.5520e-02],
          [ 7.5028e-02, -7.8743e-02, -6.6412e-02,  4.2021e-03, -5.1578e-02],
          [ 5.1529e-02,  9.2362e-03, -2.1398e-02, -2.1437e-02, -4.7125e-02]],

         [[-7.2501e-02, -4.0248e-02, -3.4060e-02, -4.5843e-02, -6.1997e-02],
          [ 7.2132e-02, -6.0532e-03, -3.4821e-02,  7.6902e-02,  2.2356e-03],
          [-2.7966e-02, -5.3776e-02, -6.6365e-03,  4.6696e-02, -2.1584e-02],
          [-6.9934e-02, -1.0891e-02,  2.6487e-02,  4.7552e-04, -2.5004e-02],
          [ 5.1597e-03, -5.6051e-02, -5.8669e-02, -7.7747e-02, -6.0225e-02]],

         [[-3.7476e-02,  1.3933e-02, -4.6468e-02,  5.2780e-02,  5.6138e-02],
          [ 7.0849e-02, -4.7803e-02, -3.7129e-02, -2.1095e-02,  3.6698e-02],
          [-7.2478e-02, -2.3833e-02,  2.8821e-03,  4.0490e-02, -6.8030e-02],
          [-7.8784e-02, -7.0211e-02, -2.6463e-02, -3.1503e-02, -1.2680e-02],
          [ 5.5113e-02,  3.3411e-02,  5.1111e-02,  5.5915e-02, -6.1831e-02]],

         [[-4.9147e-02, -6.0600e-02, -7.8963e-03,  2.3991e-02, -7.8218e-02],
          [-5.7507e-02, -2.7421e-02, -2.5888e-02,  1.2637e-02, -4.3034e-02],
          [-6.3124e-02, -6.9057e-02,  5.6438e-03,  3.2625e-02,  1.1167e-02],
          [-4.2705e-02, -3.2303e-02,  5.5996e-02, -6.1527e-02, -2.8944e-02],
          [ 2.1582e-02,  4.2828e-02,  5.6012e-02,  5.9163e-02, -5.2833e-02]],

         [[-3.9386e-03, -7.3043e-02, -5.7102e-02,  2.3391e-02,  7.5600e-02],
          [ 4.8206e-02, -3.9144e-03, -2.8480e-02,  5.4348e-02, -2.5426e-02],
          [ 2.4813e-02,  7.0907e-02,  5.5852e-02, -8.6326e-04,  9.5220e-03],
          [ 5.3540e-03, -1.3048e-02, -1.7518e-02, -1.1688e-02,  4.2103e-02],
          [ 2.3180e-03,  7.1056e-02,  2.1100e-02, -1.3906e-02,  6.2301e-02]]],


        [[[ 7.2458e-02, -4.3255e-02, -1.8478e-02,  5.0660e-02,  4.0490e-02],
          [ 6.5495e-02,  3.3672e-02,  4.0652e-02,  1.4319e-02, -4.0577e-02],
          [ 1.4689e-02,  6.0191e-03,  8.0738e-02, -2.8427e-02,  7.5639e-02],
          [-7.7613e-02, -2.3654e-04, -7.8739e-02, -7.9147e-02, -5.4791e-02],
          [-4.6359e-02, -4.4000e-02, -7.0060e-02, -6.3282e-02,  1.0501e-03]],

         [[-7.7655e-02,  3.0755e-02,  7.8193e-02, -5.9900e-02,  4.0954e-02],
          [ 3.1617e-02, -2.6465e-02,  5.8583e-02,  8.1462e-02,  2.2983e-03],
          [ 5.7186e-02, -3.4787e-02, -1.6282e-02, -4.8360e-02,  1.6764e-05],
          [-4.4135e-02,  1.5066e-02,  3.2195e-02, -2.8936e-02,  1.4467e-02],
          [-4.6183e-02,  5.7911e-03, -1.3416e-02, -3.9279e-02,  5.8995e-02]],

         [[-3.3800e-02,  1.4402e-02, -2.7597e-02, -4.6069e-02, -6.7739e-02],
          [ 5.5048e-02,  5.8227e-02,  6.4940e-02, -6.6399e-02, -5.5371e-02],
          [-8.1490e-02,  4.5180e-02, -3.3460e-02, -4.6696e-02,  4.6505e-03],
          [-7.5860e-02, -6.9557e-03,  1.6902e-02, -6.1069e-02,  1.9642e-02],
          [-5.9074e-02, -5.4762e-02,  2.7197e-02, -2.1156e-02,  2.1363e-02]],

         [[-1.9360e-02,  5.4701e-02,  6.6547e-02, -8.0681e-02,  7.5682e-02],
          [-2.2743e-02,  4.5613e-02,  5.5853e-02,  6.1339e-02, -4.1529e-02],
          [ 3.5397e-02,  3.2409e-02,  2.9896e-02, -7.7914e-02,  4.8857e-02],
          [ 1.0571e-02, -1.8501e-02, -8.0828e-02,  6.6642e-02,  3.8765e-02],
          [-5.3455e-02, -1.4353e-02,  2.6564e-02, -1.2089e-02,  2.9712e-02]],

         [[ 7.6496e-02,  5.6559e-02,  5.1197e-02, -2.9716e-02, -1.7697e-02],
          [-3.3897e-02,  3.6962e-02,  4.6940e-02,  5.9156e-02,  5.6549e-02],
          [-1.8755e-02,  3.5391e-02,  5.9375e-02, -2.9851e-02, -5.0755e-02],
          [ 7.1213e-02,  2.9275e-02, -5.0032e-02,  4.8645e-02, -5.1129e-02],
          [-3.7510e-02, -4.8853e-02,  2.8153e-02,  5.5776e-02,  7.9312e-02]],

         [[ 6.5955e-02, -1.5015e-02,  3.2332e-02, -2.1260e-02, -7.0128e-02],
          [ 7.6307e-02, -7.2524e-02, -3.2579e-02, -5.3132e-02,  5.2957e-02],
          [ 3.3710e-02, -4.2932e-02,  6.3784e-03, -3.6298e-02, -6.5088e-02],
          [ 7.0525e-02,  6.2733e-02, -7.4810e-02, -6.8870e-02,  4.3481e-02],
          [-7.9338e-02, -3.5005e-02, -1.6605e-02, -7.2584e-02, -1.3310e-02]]]]
Conv2_kernel_bias = [-0.0689,  0.0583, -0.0552]
Conv2_kernel = np.array(Conv2_kernel)
Pool2_kernel_filter_size = (2,2)                                                                                                # 2*2 down sampling

Conv1 = Convolution(reshape_test_data, Conv1_kernel, Conv1_kernel_bias)
Max_pooling1 = Max_pooling(Conv1, Pool1_kernel_filter_size)

Conv2 = Convolution(Max_pooling1, Conv2_kernel, Conv2_kernel_bias)
Max_pooling2= Max_pooling(Conv2, Pool2_kernel_filter_size)
Fully_connected = Fully_connected(Max_pooling2)