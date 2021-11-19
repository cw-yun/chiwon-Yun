import numpy as np
import activaion_file

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

    fc1_output = activaion_file.relu(fc1_output)

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

    accuracy = activaion_file.softmax(fc2_output)
    print(accuracy)

    return accuracy