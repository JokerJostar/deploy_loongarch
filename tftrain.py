import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import tensorflow_model_optimization as tfmot

# 日志输出函数
def log(message):
    print(f"[INFO] {message}")

# 步骤 1: 加载数据集
def load_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    data = np.array([float(line.strip()) for line in data], dtype=np.float32)  # 确保数据类型为 float32
    return data

def process_file(file_path):
    data = load_data_from_txt(file_path)
    label = 1 if 'AFIB' in file_path else 0
    return data, label

def load_dataset(data_dir, max_workers=4):
    file_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.txt')]
    X, y = [], []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_file, file_paths), total=len(file_paths), desc="Loading data"))
    
    for data, label in results:
        X.append(data)
        y.append(label)
    
    X = np.array(X, dtype=np.float32).reshape(-1, 1250, 1, 1)  # 更新输入形状为 (样本数, 1250, 1, 1) 并确保数据类型为 float32
    y = np.array(y).reshape(-1, 1)
    return X, y

data_dir = 'train_data/train_data'
log("Loading dataset...")
X, y = load_dataset(data_dir, max_workers=8)  # 增加 max_workers 数量以加快速度
log(f"Dataset loaded: {X.shape[0]} samples")



def simple_resnet_block(inputs, filters, kernel_size, strides, use_projection_shortcut=False):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(x)

    if use_projection_shortcut:
        shortcut = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding='same')(inputs)
    else:
        shortcut = inputs

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def simple_resnet(input_shape=(1250, 1, 1)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # 第一个卷积层和最大池化层
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 1), strides=(2, 1), padding='same')(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=(2, 1), padding='same')(x)
    
    # 保持残差块中的卷积层过滤器数量为32
    x = simple_resnet_block(x, filters=16, kernel_size=(3, 1), strides=(1, 1))
    x = simple_resnet_block(x, filters=16, kernel_size=(3, 1), strides=(2, 1), use_projection_shortcut=True)
    
    # 全局平均池化和全连接层
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)  # 将全连接层的神经元数量减少到16
    outputs = tf.keras.layers.Dense(2)(x)  # 这里不使用激活函数
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model




# Create ResNet model
model = simple_resnet()

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])



# 步骤 3: 训练模型
log("Splitting dataset...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
log(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")

# 定义早停回调
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 使用 Adam 优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 自定义回调函数以显示进度条中的损失和准确度
class TQDMProgressBar(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']
        self.progress_bar = tqdm(total=self.epochs * self.steps, desc='Training', unit='step')

    def on_train_batch_end(self, batch, logs=None):
        self.progress_bar.update(1)
        self.progress_bar.set_postfix(loss=logs['loss'], accuracy=logs['accuracy'])

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.set_postfix(epoch=epoch+1, loss=logs['loss'], accuracy=logs['accuracy'])
        self.progress_bar.update(self.steps - self.progress_bar.n % self.steps)

    def on_train_end(self, logs=None):
        self.progress_bar.close()

log("Starting training...")
progress_bar = TQDMProgressBar()
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, progress_bar])


log("Converting model to TensorFlow Lite with int8 quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()


with open('cnn_model_complex.tflite', 'wb') as f:
    f.write(tflite_model)

log("Model converted and saved as cnn_model_complex.tflite")
