import numpy as np

def Convolution(input,kernel):
    num_input, input_width, input_height = input.shape
    num_channel, kernel_width, kernel_height = kernel.shape

    new_width = input_width - kernel_width + 1
    new_height = input_height - kernel_height + 1
    conv = []
    for i in range(num_input):
        for j in range(num_channel):
            for k in range(new_width):
                for l in range(new_height):
                    conv.append((input[i,k:k + kernel_width, l:l + kernel_height] * kernel[j]).sum())

    global num_all_channel
    num_all_channel *= num_channel
    conv = np.array(conv).reshape(num_all_channel,new_width,new_height)
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
reshape_test_data = test_data[1:].reshape(1,28,28)
#print(reshape_test_data.shape)

num_all_channel = 1

Conv1_kernel = [[[0.01,0,0.01,0,0.01],[0,0.01,0,0.01,0],[0.01,0,0.01,0,0.01],[0,0.01,0,0.01,0],[0.01,0,0.01,0,0.01]],            # 6 channels, 5*5 kernel
                [[0.011,0,0.011,0,0.011],[0,0.011,0,0.011,0],[0.011,0,0.011,0,0.011],[0,0.011,0,0.011,0],[0.011,0,0.011,0,0.011]],
                [[0.012,0,0.012,0,0.012],[0,0.012,0,0.012,0],[0.012,0,0.012,0,0.012],[0,0.012,0,0.012,0],[0.012,0,0.012,0,0.012]],
                [[0.013,0,0.013,0,0.013],[0,0.013,0,0.013,0],[0.013,0,0.013,0,0.013],[0,0.013,0,0.013,0],[0.013,0,0.013,0,0.013]],
                [[0.014,0,0.014,0,0.014],[0,0.014,0,0.014,0],[0.014,0,0.014,0,0.014],[0,0.014,0,0.014,0],[0.014,0,0.014,0,0.014]],
                [[0.015,0,0.015,0,0.015],[0,0.015,0,0.015,0],[0.015,0,0.015,0,0.015],[0,0.015,0,0.015,0],[0.015,0,0.015,0,0.015]]]
Conv1_kernel = np.array(Conv1_kernel)
Pool1_kernel_channel_size = (2,2)                                                                                                # 2*2 down sampling

Conv2_kernel = [[[0.01,0,0.01,0,0.01],[0,0.01,0,0.01,0],[0.01,0,0.01,0,0.01],[0,0.01,0,0.01,0],[0.01,0,0.01,0,0.01]],            # 3 channels, 5*5 kernel
                [[0.011,0,0.011,0,0.011],[0,0.011,0,0.011,0],[0.011,0,0.011,0,0.011],[0,0.011,0,0.011,0],[0.011,0,0.011,0,0.011]],
                [[0.012,0,0.012,0,0.012],[0,0.012,0,0.012,0],[0.012,0,0.012,0,0.012],[0,0.012,0,0.012,0],[0.012,0,0.012,0,0.012]]]
Conv2_kernel = np.array(Conv2_kernel)
Pool2_kernel_channel_size = (2,2)                                                                                                # 2*2 down sampling

Conv1 = Convolution(reshape_test_data,Conv1_kernel)
Max_pooling1 = Max_pooling(Conv1,Pool1_kernel_channel_size)

Conv2 = Convolution(Max_pooling1,Conv2_kernel)
Max_pooling2= Max_pooling(Conv2,Pool2_kernel_channel_size)
Fully_connected = Fully_connected(Max_pooling2)