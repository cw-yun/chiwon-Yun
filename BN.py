def BatchNormalization(input, gamma, beta, mean, var):
    num_channel, num_input, input_width, input_height = input.shape
    eps = 0.00001

    for i in range(num_channel):
        for j in range(num_input):
            for k in range(input_height):
                for l in range(input_width):
                    input[i][j][k][l] = (input[i][j][k][l] - mean[j]) / ((var[j] + eps) ** (1 / 2)) * gamma[j] + beta[j]

    output = input
    print('----------------------------------BN layer------------------------------------')
    print(output.shape)

    return output