import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin

# 定义模型创建函数
def create_model(learning_rate=0.01, neurons=64):
    model = Sequential()
    model.add(Dense(neurons, input_dim=4, activation='relu'))  # 输入层
    model.add(Dense(3, activation='softmax'))  # 输出层
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 自定义 KerasClassifier
# 就自带的keras插件它升级不了又不让用我们就自己手动搭建了如下
#看不懂但反正能用
class MyKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn=create_model, epochs=10, batch_size=32, learning_rate=0.01, neurons=64, verbose=0):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.neurons = neurons
        self.verbose = verbose
        self.model_ = None

    def fit(self, X, y):
        self.model_ = self.build_fn(learning_rate=self.learning_rate, neurons=self.neurons)
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        return np.argmax(self.model_.predict(X), axis=1)

    def score(self, X, y):
        loss, accuracy = self.model_.evaluate(X, y, verbose=self.verbose)
        return accuracy

# 读取 CSV 文件
data = pd.read_csv('your_file.csv')

# 假设特征数据位于前几列，目标列是最后一列
X = data.iloc[:, :-1].values  # 特征列
y = data.iloc[:, -1].values   # 目标列

scaler = StandardScaler()  # 你可以选择使用 Normalizer() 代替 StandardScaler
X = scaler.fit_transform(X)  # 标准化特征数据

# 进行标签编码和 one-hot 编码
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)  # 转换为 one-hot 编码

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 KerasClassifier 实例
model = MyKerasClassifier()

# 定义超参数网格
param_grid = {
    'epochs': [50, 100],
    'batch_size': [5, 10],
    'learning_rate': [0.01, 0.001],
    'neurons': [32, 64]
}

# 进行超参数搜索
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# 输出最佳参数和最佳得分
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# 评估最佳模型
best_model = grid_result.best_estimator_
test_accuracy = best_model.score(X_test, y_test)

print(f"Best model test accuracy: {test_accuracy}")
