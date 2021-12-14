import numpy as np
from PIL import Image
from torchvision import transforms
import ast
import conv_with_padding_stride
import BN
import activation_file
import linear_function
import time
import re
import torch
import torch.nn.functional as F

torch.set_printoptions(precision=30) # 소수점 n째자리까지만 계산
np.set_printoptions(precision=30) # 소수점 n째자리까지만 계산
np.set_printoptions(threshold=np.inf) # 모든 배열의 수 출력

input = Image.open('dog.jpg')

preprocess = transforms.Compose([
    transforms.Resize(256, interpolation = Image.BILINEAR), # bilinear mode : 92.848% / nearest mode : 84.169%
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
input = preprocess(input)
input = np.array(input).reshape(3, 1, 224, 224) # size : (channel, num_input, width, height)

# weight, bias 값 텍스트 파일로부터 불러오기
f = open('MobileNetV2_weight_bias.txt','r')
lines = f.read() # string 형태로 값 읽기
lines = re.findall('tensor\(([^)]+)', lines) # 'tensor(' 이후부터 ')'까지만 입력받음

for i in range(0, len(lines)):
    lines[i] = ast.literal_eval(lines[i])  # ast 라이브러리를 이용하여 string 형태의 문자열을 list 형태로 변환

# weight, bias, mean, var 값 대입
for i in range(0, int((len(lines) - 2 - 52) / 5)): # 전체 lines[] 중에 마지막 2개 dropout, linear layer는 제외(-2) / 또한 num_batches_tracked 제외(-52) / 한 for문에 5개의 다른 입력을 대입하므로 5로 나누기
    globals()['Conv2d_{}_weight'.format(3 * i + 1)] = np.array(lines[6 * i])
    globals()['BN_{}_weight'.format(3 * i + 2)] = np.array(lines[6 * i + 1])
    globals()['BN_{}_bias'.format(3 * i + 2)] = np.array(lines[6 * i + 2])
    globals()['BN_{}_mean'.format(3 * i + 2)] = np.array(lines[6 * i + 3])
    globals()['BN_{}_var'.format(3 * i + 2)] = np.array(lines[6 * i + 4])
Linear_weight, Linear_bias = np.array(lines[312]), np.array(lines[313])

start = time.time() # 시간 측정 시작

# layer 쌓기

Conv2d_1_output = conv_with_padding_stride.Convolution(input, Conv2d_1_weight, (2,2), 1, None)
BN_2_output = BN.BatchNormalization(Conv2d_1_output, BN_2_weight, BN_2_bias, BN_2_mean, BN_2_var)
ReLU6_3_output = activation_file.relu6(BN_2_output)
Conv2d_4_output = conv_with_padding_stride.Convolution(ReLU6_3_output, Conv2d_4_weight, (1,1), 1, None)
BN_5_output = BN.BatchNormalization(Conv2d_4_output, BN_5_weight, BN_5_bias, BN_5_mean, BN_5_var)
ReLU6_6_output = activation_file.relu6(BN_5_output)
Conv2d_7_output = conv_with_padding_stride.Convolution(ReLU6_6_output, Conv2d_7_weight, (1,1), 0, None)
BN_8_output = BN.BatchNormalization(Conv2d_7_output, BN_8_weight, BN_8_bias, BN_8_mean, BN_8_var)
Conv2d_10_output = conv_with_padding_stride.Convolution(BN_8_output, Conv2d_10_weight, (1,1), 0, None)
BN_11_output = BN.BatchNormalization(Conv2d_10_output, BN_11_weight, BN_11_bias, BN_11_mean, BN_11_var)
ReLU6_12_output = activation_file.relu6(BN_11_output)
Conv2d_13_output = conv_with_padding_stride.Convolution(ReLU6_12_output, Conv2d_13_weight, (2,2), 1, None)
BN_14_output = BN.BatchNormalization(Conv2d_13_output, BN_14_weight, BN_14_bias, BN_14_mean, BN_14_var)
ReLU6_15_output = activation_file.relu6(BN_14_output)
Conv2d_16_output = conv_with_padding_stride.Convolution(ReLU6_15_output, Conv2d_16_weight, (1,1), 0, None)
BN_17_output = BN.BatchNormalization(Conv2d_16_output, BN_17_weight, BN_17_bias, BN_17_mean, BN_17_var)
Conv2d_19_output = conv_with_padding_stride.Convolution(BN_17_output, Conv2d_19_weight, (1,1), 0, None)
BN_20_output = BN.BatchNormalization(Conv2d_19_output, BN_20_weight, BN_20_bias, BN_20_mean, BN_20_var)
ReLU6_21_output = activation_file.relu6(BN_20_output)
Conv2d_22_output = conv_with_padding_stride.Convolution(ReLU6_21_output, Conv2d_22_weight, (1,1), 1, None)
BN_23_output = BN.BatchNormalization(Conv2d_22_output, BN_23_weight, BN_23_bias, BN_23_mean, BN_23_var)
ReLU6_24_output = activation_file.relu6(BN_23_output)
Conv2d_25_output = conv_with_padding_stride.Convolution(ReLU6_24_output, Conv2d_25_weight, (1,1), 0, None)
BN_26_output = BN.BatchNormalization(Conv2d_25_output, BN_26_weight, BN_26_bias, BN_26_mean, BN_26_var)

BN_26_output = BN_17_output + BN_26_output # Inverted Residual 구조에서 input의 크기와 output의 크기가 같은 경우에는 skip connection을 추가(최종 output = Inverted Residual 구조의 input + Inverted Residual 구조의 output)

Conv2d_28_output = conv_with_padding_stride.Convolution(BN_26_output, Conv2d_28_weight, (1,1), 0, None)
BN_29_output = BN.BatchNormalization(Conv2d_28_output, BN_29_weight, BN_29_bias, BN_29_mean, BN_29_var)
ReLU6_30_output = activation_file.relu6(BN_29_output)
Conv2d_31_output = conv_with_padding_stride.Convolution(ReLU6_30_output, Conv2d_31_weight, (2,2), 1, None)
BN_32_output = BN.BatchNormalization(Conv2d_31_output, BN_32_weight, BN_32_bias, BN_32_mean, BN_32_var)
ReLU6_33_output = activation_file.relu6(BN_32_output)
Conv2d_34_output = conv_with_padding_stride.Convolution(ReLU6_33_output, Conv2d_34_weight, (1,1), 0, None)
BN_35_output = BN.BatchNormalization(Conv2d_34_output, BN_35_weight, BN_35_bias, BN_35_mean, BN_35_var)
Conv2d_37_output = conv_with_padding_stride.Convolution(BN_35_output, Conv2d_37_weight, (1,1), 0, None)
BN_38_output = BN.BatchNormalization(Conv2d_37_output, BN_38_weight, BN_38_bias, BN_38_mean, BN_38_var)
ReLU6_39_output = activation_file.relu6(BN_38_output)
Conv2d_40_output = conv_with_padding_stride.Convolution(ReLU6_39_output, Conv2d_40_weight, (1,1), 1, None)
BN_41_output = BN.BatchNormalization(Conv2d_40_output, BN_41_weight, BN_41_bias, BN_41_mean, BN_41_var)
ReLU6_42_output = activation_file.relu6(BN_41_output)
Conv2d_43_output = conv_with_padding_stride.Convolution(ReLU6_42_output, Conv2d_43_weight, (1,1), 0, None)
BN_44_output = BN.BatchNormalization(Conv2d_43_output, BN_44_weight, BN_44_bias, BN_44_mean, BN_44_var)

BN_44_output = BN_35_output + BN_44_output

Conv2d_46_output = conv_with_padding_stride.Convolution(BN_44_output, Conv2d_46_weight, (1,1), 0, None)
BN_47_output = BN.BatchNormalization(Conv2d_46_output, BN_47_weight, BN_47_bias, BN_47_mean, BN_47_var)
ReLU6_48_output = activation_file.relu6(BN_47_output)
Conv2d_49_output = conv_with_padding_stride.Convolution(ReLU6_48_output, Conv2d_49_weight, (1,1), 1, None)
BN_50_output = BN.BatchNormalization(Conv2d_49_output, BN_50_weight, BN_50_bias, BN_50_mean, BN_50_var)
ReLU6_51_output = activation_file.relu6(BN_50_output)
Conv2d_52_output = conv_with_padding_stride.Convolution(ReLU6_51_output, Conv2d_52_weight, (1,1), 0, None)
BN_53_output = BN.BatchNormalization(Conv2d_52_output, BN_53_weight, BN_53_bias, BN_53_mean, BN_53_var)

BN_53_output = BN_44_output + BN_53_output

Conv2d_55_output = conv_with_padding_stride.Convolution(BN_53_output, Conv2d_55_weight, (1,1), 0, None)
BN_56_output = BN.BatchNormalization(Conv2d_55_output, BN_56_weight, BN_56_bias, BN_56_mean, BN_56_var)
ReLU6_57_output = activation_file.relu6(BN_56_output)
Conv2d_58_output = conv_with_padding_stride.Convolution(ReLU6_57_output, Conv2d_58_weight, (2,2), 1, None)
BN_59_output = BN.BatchNormalization(Conv2d_58_output, BN_59_weight, BN_59_bias, BN_59_mean, BN_59_var)
ReLU6_60_output = activation_file.relu6(BN_59_output)
Conv2d_61_output = conv_with_padding_stride.Convolution(ReLU6_60_output, Conv2d_61_weight, (1,1), 0, None)
BN_62_output = BN.BatchNormalization(Conv2d_61_output, BN_62_weight, BN_62_bias, BN_62_mean, BN_62_var)
Conv2d_64_output = conv_with_padding_stride.Convolution(BN_62_output, Conv2d_64_weight, (1,1), 0, None)
BN_65_output = BN.BatchNormalization(Conv2d_64_output, BN_65_weight, BN_65_bias, BN_65_mean, BN_65_var)
ReLU6_66_output = activation_file.relu6(BN_65_output)
Conv2d_67_output = conv_with_padding_stride.Convolution(ReLU6_66_output, Conv2d_67_weight, (1,1), 1, None)
BN_68_output = BN.BatchNormalization(Conv2d_67_output, BN_68_weight, BN_68_bias, BN_68_mean, BN_68_var)
ReLU6_69_output = activation_file.relu6(BN_68_output)
Conv2d_70_output = conv_with_padding_stride.Convolution(ReLU6_69_output, Conv2d_70_weight, (1,1), 0, None)
BN_71_output = BN.BatchNormalization(Conv2d_70_output, BN_71_weight, BN_71_bias, BN_71_mean, BN_71_var)

BN_71_output = BN_62_output + BN_71_output

Conv2d_73_output = conv_with_padding_stride.Convolution(BN_71_output, Conv2d_73_weight, (1,1), 0, None)
BN_74_output = BN.BatchNormalization(Conv2d_73_output, BN_74_weight, BN_74_bias, BN_74_mean, BN_74_var)
ReLU6_75_output = activation_file.relu6(BN_74_output)
Conv2d_76_output = conv_with_padding_stride.Convolution(ReLU6_75_output, Conv2d_76_weight, (1,1), 1, None)
BN_77_output = BN.BatchNormalization(Conv2d_76_output, BN_77_weight, BN_77_bias, BN_77_mean, BN_77_var)
ReLU6_78_output = activation_file.relu6(BN_77_output)
Conv2d_79_output = conv_with_padding_stride.Convolution(ReLU6_78_output, Conv2d_79_weight, (1,1), 0, None)
BN_80_output = BN.BatchNormalization(Conv2d_79_output, BN_80_weight, BN_80_bias, BN_80_mean, BN_80_var)

BN_80_output = BN_71_output + BN_80_output

Conv2d_82_output = conv_with_padding_stride.Convolution(BN_80_output, Conv2d_82_weight, (1,1), 0, None)
BN_83_output = BN.BatchNormalization(Conv2d_82_output, BN_83_weight, BN_83_bias, BN_83_mean, BN_83_var)
ReLU6_84_output = activation_file.relu6(BN_83_output)
Conv2d_85_output = conv_with_padding_stride.Convolution(ReLU6_84_output, Conv2d_85_weight, (1,1), 1, None)
BN_86_output = BN.BatchNormalization(Conv2d_85_output, BN_86_weight, BN_86_bias, BN_86_mean, BN_86_var)
ReLU6_87_output = activation_file.relu6(BN_86_output)
Conv2d_88_output = conv_with_padding_stride.Convolution(ReLU6_87_output, Conv2d_88_weight, (1,1), 0, None)
BN_89_output = BN.BatchNormalization(Conv2d_88_output, BN_89_weight, BN_89_bias, BN_89_mean, BN_89_var)

BN_89_output = BN_80_output + BN_89_output

Conv2d_91_output = conv_with_padding_stride.Convolution(BN_89_output, Conv2d_91_weight, (1,1), 0, None)
BN_92_output = BN.BatchNormalization(Conv2d_91_output, BN_92_weight, BN_92_bias, BN_92_mean, BN_92_var)
ReLU6_93_output = activation_file.relu6(BN_92_output)
Conv2d_94_output = conv_with_padding_stride.Convolution(ReLU6_93_output, Conv2d_94_weight, (1,1), 1, None)
BN_95_output = BN.BatchNormalization(Conv2d_94_output, BN_95_weight, BN_95_bias, BN_95_mean, BN_95_var)
ReLU6_96_output = activation_file.relu6(BN_95_output)
Conv2d_97_output = conv_with_padding_stride.Convolution(ReLU6_96_output, Conv2d_97_weight, (1,1), 0, None)
BN_98_output = BN.BatchNormalization(Conv2d_97_output, BN_98_weight, BN_98_bias, BN_98_mean, BN_98_var)
Conv2d_100_output = conv_with_padding_stride.Convolution(BN_98_output, Conv2d_100_weight, (1,1), 0, None)
BN_101_output = BN.BatchNormalization(Conv2d_100_output, BN_101_weight, BN_101_bias, BN_101_mean, BN_101_var)
ReLU6_102_output = activation_file.relu6(BN_101_output)
Conv2d_103_output = conv_with_padding_stride.Convolution(ReLU6_102_output, Conv2d_103_weight, (1,1), 1, None)
BN_104_output = BN.BatchNormalization(Conv2d_103_output, BN_104_weight, BN_104_bias, BN_104_mean, BN_104_var)
ReLU6_105_output = activation_file.relu6(BN_104_output)
Conv2d_106_output = conv_with_padding_stride.Convolution(ReLU6_105_output, Conv2d_106_weight, (1,1), 0, None)
BN_107_output = BN.BatchNormalization(Conv2d_106_output, BN_107_weight, BN_107_bias, BN_107_mean, BN_107_var)

BN_107_output = BN_98_output + BN_107_output

Conv2d_109_output = conv_with_padding_stride.Convolution(BN_107_output, Conv2d_109_weight, (1,1), 0, None)
BN_110_output = BN.BatchNormalization(Conv2d_109_output, BN_110_weight, BN_110_bias, BN_110_mean, BN_110_var)
ReLU6_111_output = activation_file.relu6(BN_110_output)
Conv2d_112_output = conv_with_padding_stride.Convolution(ReLU6_111_output, Conv2d_112_weight, (1,1), 1, None)
BN_113_output = BN.BatchNormalization(Conv2d_112_output, BN_113_weight, BN_113_bias, BN_113_mean, BN_113_var)
ReLU6_114_output = activation_file.relu6(BN_113_output)
Conv2d_115_output = conv_with_padding_stride.Convolution(ReLU6_114_output, Conv2d_115_weight, (1,1), 0, None)
BN_116_output = BN.BatchNormalization(Conv2d_115_output, BN_116_weight, BN_116_bias, BN_116_mean, BN_116_var)

BN_116_output = BN_107_output + BN_116_output

Conv2d_118_output = conv_with_padding_stride.Convolution(BN_116_output, Conv2d_118_weight, (1,1), 0, None)
BN_119_output = BN.BatchNormalization(Conv2d_118_output, BN_119_weight, BN_119_bias, BN_119_mean, BN_119_var)
ReLU6_120_output = activation_file.relu6(BN_119_output)
Conv2d_121_output = conv_with_padding_stride.Convolution(ReLU6_120_output, Conv2d_121_weight, (2,2), 1, None)
BN_122_output = BN.BatchNormalization(Conv2d_121_output, BN_122_weight, BN_122_bias, BN_122_mean, BN_122_var)
ReLU6_123_output = activation_file.relu6(BN_122_output)
Conv2d_124_output = conv_with_padding_stride.Convolution(ReLU6_123_output, Conv2d_124_weight, (1,1), 0, None)
BN_125_output = BN.BatchNormalization(Conv2d_124_output, BN_125_weight, BN_125_bias, BN_125_mean, BN_125_var)
Conv2d_127_output = conv_with_padding_stride.Convolution(BN_125_output, Conv2d_127_weight, (1,1), 0, None)
BN_128_output = BN.BatchNormalization(Conv2d_127_output, BN_128_weight, BN_128_bias, BN_128_mean, BN_128_var)
ReLU6_129_output = activation_file.relu6(BN_128_output)
Conv2d_130_output = conv_with_padding_stride.Convolution(ReLU6_129_output, Conv2d_130_weight, (1,1), 1, None)
BN_131_output = BN.BatchNormalization(Conv2d_130_output, BN_131_weight, BN_131_bias, BN_131_mean, BN_131_var)
ReLU6_132_output = activation_file.relu6(BN_131_output)
Conv2d_133_output = conv_with_padding_stride.Convolution(ReLU6_132_output, Conv2d_133_weight, (1,1), 0, None)
BN_134_output = BN.BatchNormalization(Conv2d_133_output, BN_134_weight, BN_134_bias, BN_134_mean, BN_134_var)

BN_134_output = BN_125_output + BN_134_output

Conv2d_136_output = conv_with_padding_stride.Convolution(BN_134_output, Conv2d_136_weight, (1,1), 0, None)
BN_137_output = BN.BatchNormalization(Conv2d_136_output, BN_137_weight, BN_137_bias, BN_137_mean, BN_137_var)
ReLU6_138_output = activation_file.relu6(BN_137_output)
Conv2d_139_output = conv_with_padding_stride.Convolution(ReLU6_138_output, Conv2d_139_weight, (1,1), 1, None)
BN_140_output = BN.BatchNormalization(Conv2d_139_output, BN_140_weight, BN_140_bias, BN_140_mean, BN_140_var)
ReLU6_141_output = activation_file.relu6(BN_140_output)
Conv2d_142_output = conv_with_padding_stride.Convolution(ReLU6_141_output, Conv2d_142_weight, (1,1), 0, None)
BN_143_output = BN.BatchNormalization(Conv2d_142_output, BN_143_weight, BN_143_bias, BN_143_mean, BN_143_var)

BN_143_output = BN_134_output + BN_143_output

Conv2d_145_output = conv_with_padding_stride.Convolution(BN_143_output, Conv2d_145_weight, (1,1), 0, None)
BN_146_output = BN.BatchNormalization(Conv2d_145_output, BN_146_weight, BN_146_bias, BN_146_mean, BN_146_var)
ReLU6_147_output = activation_file.relu6(BN_146_output)
Conv2d_148_output = conv_with_padding_stride.Convolution(ReLU6_147_output, Conv2d_148_weight, (1,1), 1, None)
BN_149_output = BN.BatchNormalization(Conv2d_148_output, BN_149_weight, BN_149_bias, BN_149_mean, BN_149_var)
ReLU6_150_output = activation_file.relu6(BN_149_output)
Conv2d_151_output = conv_with_padding_stride.Convolution(ReLU6_150_output, Conv2d_151_weight, (1,1), 0, None)
BN_152_output = BN.BatchNormalization(Conv2d_151_output, BN_152_weight, BN_152_bias, BN_152_mean, BN_152_var)
Conv2d_154_output = conv_with_padding_stride.Convolution(BN_152_output, Conv2d_154_weight, (1,1), 0, None)
BN_155_output = BN.BatchNormalization(Conv2d_154_output, BN_155_weight, BN_155_bias, BN_155_mean, BN_155_var)
ReLU6_156_output = activation_file.relu6(BN_155_output)

# 최종결과를 tensor 형태로 변환하고 이를 Avg_pooling layer와 Flatten에 적용(Dropout은 train에만 적용하므로 여기서는 적용하지 않음)
ReLU6_156_output = torch.tensor(ReLU6_156_output)
Avg_pooling_output = F.adaptive_avg_pool2d(ReLU6_156_output, (1,1)) # avg_pooling layer
Flatten_output = torch.flatten(Avg_pooling_output, 1) # flatten layer
Flatten_output = np.array(Flatten_output)
print('--------------------------Avg_pool & Flatten layer----------------------------')
print(Flatten_output.shape)

# Flatten layer의 결과를 numpy 형태로 변환하고 이를 Linear함수에 대입
Linear_output = linear_function.Linear(Flatten_output, Linear_weight, Linear_bias)

# softmax
final_output = activation_file.softmax(Linear_output)
print('-------------------------------softmax layer------------------------------------')
print(final_output.shape)

print('------------------------------------Time----------------------------------------')
print("걸리는 시간 :", time.time() - start) # 걸리는 시간 출력

# txt 파일에서 가장 높은 정확도를 나타내는 5개 출력
print('-------------------------------Final Accuracy-----------------------------------')
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
final_output = torch.tensor(final_output)
top5_prob, top5_catid = torch.topk(final_output, 5)

for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())