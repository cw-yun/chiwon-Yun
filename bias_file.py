def Bias(conv_sum, bias):
    num_channel, num_input, input_width, input_height = conv_sum.shape

    for i in range(num_channel):
            for j in range(num_input):
                for k in range(input_width):
                    for l in range(input_height):
                        conv_sum[i][j][k][l] += bias[j]
    return conv_sum