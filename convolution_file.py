import numpy as np
import matplotlib.pyplot as plt
import bias_file

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
        conv_sum = bias_file.Bias(conv_sum, bias)

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
        all_filter_sum = bias_file.Bias(all_filter_sum, bias)

        print('--------------------------------conv layer------------------------------------')
        print(all_filter_sum.shape)
        #print(all_filter_sum)

        # conv layer 이미지 출력
        plt.imshow(all_filter_sum[0][0], cmap = 'Greys')
        plt.show()
        return all_filter_sum