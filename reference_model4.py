'''
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 1. MNIST 데이터셋 불러오기
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 2. 데이터 전처리하기
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

# 3. 합성곱 신경망 구성하기
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.summary() # 위까지 구성한 신경망에 대한 정보 출력

# 4. Dense 층 추가하기
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# 5. 모델 컴파일하기
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. 훈련하기
model.fit(train_images, train_labels, epochs=3)

# 7. 모델 평가하기
loss, acc = model.evaluate(test_images, test_labels, verbose=2)

weights = model.get_weights()
print(weights)
print(weights[0].shape)
'''



import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

# 1. MNIST 데이터셋 불러오기
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 2. 데이터 전처리하기
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

# 3. 합성곱 신경망 구성하기
model = models.Sequential()
model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(18, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))

#model.summary()

# 4. Dense 층 추가하기
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()      # 위까지 구성한 신경망에 대한 정보 출력

# 5. 모델 컴파일하기
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. 훈련하기
model.fit(train_images, train_labels, epochs=3)
model.save('reference_model4')
model.save_weights('reference_model4_weights')
# 7. 모델 평가하기
loss, acc = model.evaluate(test_images, test_labels, verbose=2)

weights = model.get_weights()
#print(weights)
print(weights[0].shape)
#print(weights[0])
print(weights[1].shape)
#print(weights[1])
print(weights[2].shape)
np.set_printoptions(threshold=np.inf)
print(weights[2])
print(weights[3].shape)
#print(weights[3])
print(weights[4].shape)
#print(weights[4])
print(weights[5].shape)
#print(weights[5])
print(weights[6].shape)
#print(weights[6])
print(weights[7].shape)
#print(weights[7])

#np.savetxt('weights[0].csv', weights[0])



'''
from keras.models import load_model
model = load_model('reference_model4')
weights = model.get_weights()
print(weights[0].shape)
print(weights[0])
print(weights[1].shape)
#print(weights[1])
print(weights[2].shape)
#print(weights[2])
print(weights[3].shape)
#print(weights[3])
print(weights[4].shape)
#print(weights[4])
print(weights[5].shape)
#print(weights[5])
print(weights[6].shape)
#print(weights[6])
print(weights[7].shape)
#print(weights[7])
'''
