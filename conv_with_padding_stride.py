import numpy as np
import matplotlib.pyplot as plt
import bias_file

def Convolution(input, kernel, stride, padding, bias):
    # padding
    if padding > 0:
        input = np.pad(input, [(0,0), (0,0), (padding, padding), (padding, padding)], mode = 'constant', constant_values = 0)

    num_channel, num_input, input_width, input_height = input.shape
    num_filter, input_node, kernel_width, kernel_height = kernel.shape
    new_width = input_width - kernel_width + 1
    new_height = input_height - kernel_height + 1
    conv_sum = []

    # input_node가 1인 경우
    if input_node == 1:
        for i in range(num_filter):
            for j in range(input_node):
                conv = []
                for k in range(0, new_height, stride[1]):
                    for l in range(0, new_width, stride[0]):
                        conv.append((input[j, i, k:k + kernel_height, l:l + kernel_width] * kernel[i][j]).sum())
                conv_sum.append(conv)

        conv_sum = np.array(conv_sum).reshape(-1, num_filter, int(new_width / stride[0]), int(new_height / stride[1]))

        # add bias
        if bias != None:
            conv_sum = bias_file.Bias(conv_sum, bias)

        print('--------------------------------conv layer------------------------------------')
        print(conv_sum.shape)
        #print(conv_sum)

        '''
        # conv layer 이미지 출력
        plt.imshow(conv_sum[0][0], cmap = 'Greys')
        plt.show()
        '''
        return conv_sum

    # input_node > 1 인 경우(sum을 하는 동작이 필요)
    elif input_node > 1:
        if num_channel == input_node:
            for i in range(num_input):
                for j in range(num_filter):
                    for k in range(num_channel):
                        conv = []
                        for l in range(0, new_height, stride[1]):
                            for m in range(0, new_width, stride[0]):
                                conv.append((input[k, i, l:l + kernel_height, m:m + kernel_width] * kernel[j][k]).sum())
                        conv_sum.append(conv)

        elif num_input == input_node:
            for i in range(num_filter):
                for j in range(input_node):
                    conv = []
                    for k in range(0, new_height, stride[1]):
                        for l in range(0, new_width, stride[0]):
                            conv.append((input[0, j, k:k + kernel_height, l:l + kernel_width] * kernel[i][j]).sum())
                    conv_sum.append(conv)

            conv_sum = np.array(conv_sum).reshape(num_filter, input_node, new_width * new_height)


        conv_sum = np.array(conv_sum).reshape(num_filter, input_node, int(new_width / stride[0]) * int(new_height / stride[1]))
        filter_sum = 0
        all_filter_sum = []

        # rgb 3개의 채널이지만 1개의 채널로 sum
        for i in range(num_filter):
            for j in range(int(new_width / stride[0]) * int(new_height / stride[1])):
                for k in range(input_node):
                    filter_sum += conv_sum[i][k][j]
                all_filter_sum.append(filter_sum)
                filter_sum = 0

        all_filter_sum = np.array(all_filter_sum).reshape(-1, num_filter, int(new_width / stride[0]), int(new_height / stride[1]))

        # add bias
        if bias != None:
            all_filter_sum = bias_file.Bias(all_filter_sum, bias)

        print('--------------------------------conv layer------------------------------------')
        print(all_filter_sum.shape)
        #print(all_filter_sum)

        '''
        # conv layer 이미지 출력
        plt.imshow(all_filter_sum[0][0], cmap = 'Greys')
        plt.show()
        '''
        return all_filter_sum
