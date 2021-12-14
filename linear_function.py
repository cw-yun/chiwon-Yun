import numpy as np

def Linear(input, weight, bias):
    num_channel, num_input = input.shape
    num_output, num_input = weight.shape

    linear_output = []
    sum = 0

    for i in range(num_channel):
        for j in range(num_output):
            for k in range(num_input):
                sum += input[i][k] * weight[j][k]
            linear_output.append(sum)
            sum = 0

    # Bias
    for i in range(num_output):
        linear_output[i] = linear_output[i] + bias[i]

    linear_output = np.array(linear_output)
    print('---------------------------------Linear layer-----------------------------------')
    print(linear_output.shape)

    return linear_output

