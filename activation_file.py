import numpy as np

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    out = exp_x / sum_exp_x
    return out

def relu6(x):
    x = np.minimum(np.maximum(0,x), 6)
    print('--------------------------------ReLU6 layer-----------------------------------')
    print(x.shape)
    return x
