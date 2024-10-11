import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理：归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

# 构建模型
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),        # 将28x28的图像展平为784维向量
    layers.Dense(128, activation='relu'),        # 全连接层，128个神经元，激活函数为ReLU
    layers.Dropout(0.2),                         # Dropout层，防止过拟合
    layers.Dense(10, activation='softmax')       # 输出层，10个神经元，对应10个类别
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 查看模型结构
model.summary()

# 训练模型
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\n测试准确率:', test_acc)

# 绘制训练和验证的准确率和损失
plt.figure(figsize=(12, 4))

# 准确率
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率')
plt.legend()

# 损失
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()

plt.show()
