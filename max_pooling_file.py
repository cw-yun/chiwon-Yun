import numpy as np
import activaion_file
import matplotlib.pyplot as plt

def Max_pooling(input,kernel_size):
    num_channel, num_input, input_width, input_height = input.shape
    input = activaion_file.relu(input)
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