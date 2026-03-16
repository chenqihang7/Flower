import tensorflow as tf
import matplotlib.pyplot as plt
import os
import keras_tuner as kt


# 数据加载函数，按8:2的比例加载花卉数据，并单独加载测试集
def data_load(data_dir, test_dir, img_height, img_width, batch_size):
    """
    加载数据集并按训练集和测试集划分。
    参数：
        data_dir: 训练数据集目录路径。
        test_dir: 测试数据集目录路径。
        img_height: 图像高度。
        img_width: 图像宽度。
        batch_size: 每批次加载的图像数量。
    返回：
        train_ds: 训练数据集。
        test_ds: 测试数据集。
        class_names: 类别名称列表。
    """
    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # 加载测试集
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # 获取类别名称
    class_names = train_ds.class_names

    return train_ds, test_ds, class_names


# 模型加载函数
def model_load(IMG_SHAPE=(224, 224, 3), is_transfer=False, dropout_rate=0.5):
    """
    加载和构建模型。
    参数：
        IMG_SHAPE: 输入图像的形状。
        is_transfer: 是否使用迁移学习。
        dropout_rate: Dropout比例。
    返回：
        model: 构建好的模型。
    """
    if is_transfer:
        # 加载预训练模型 MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False

        # 搭建模型
        model = tf.keras.models.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(5, activation='softmax')  # 输出层，5分类任务
        ])
    else:
        # 搭建自定义卷积神经网络
        model = tf.keras.models.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),#缩放输入数据，使像素值范围在[0,1]之间
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), #卷积层
            tf.keras.layers.MaxPooling2D(2, 2),  # 池化层
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), #卷积层
            tf.keras.layers.MaxPooling2D(2, 2), # 池化层
            tf.keras.layers.Flatten(), #展平层
            tf.keras.layers.Dense(128, activation='relu'), # 全连接层
            tf.keras.layers.Dropout(dropout_rate), # Dropout层 防止过拟合
            tf.keras.layers.Dense(5, activation='softmax')  # 输出层，5分类任务
        ])

    # 编译模型
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 展示训练过程的曲线
def show_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Test Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Test Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Test Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Test Loss')
    plt.xlabel('epoch')
    plt.show()


# 超参数调优函数
def model_builder(hp):
    """
    定义超参数搜索空间。
    """
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.3, step=0.1)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='LOG')
    batch_size = hp.Choice('batch_size', values=[16, 32])
    is_transfer = False



    # 构建模型
    model = model_load(IMG_SHAPE=(224, 224, 3),
                       is_transfer=is_transfer,
                       dropout_rate=dropout_rate)

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 模型训练函数
def train_with_tuner(epochs=10):  # 设置为较少的epoch
    """
    使用 Keras Tuner 进行超参数调优训练。
    """
    # 确保模型保存目录存在
    os.makedirs("models", exist_ok=True)

    # 数据加载
    train_ds, test_ds, class_names = data_load(
        "F:/AI/Flower-master/Flower-master/newdata/train",
        "F:/AI/Flower-master/Flower-master/newdata/test",
        224,
        224,
        32
    )

    # 创建 Keras Tuner 对象
    tuner = kt.Hyperband(
        model_builder,
        objective='val_accuracy',
        max_epochs=epochs,  # 设置最大训练轮数为15
        hyperband_iterations=1,
        directory='tuner_dir',
        project_name='flower_classification'
    )

    # 启动超参数优化
    tuner.search(train_ds, validation_data=test_ds, epochs=epochs)

    # 输出最好的超参数组合
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best dropout_rate: {best_hps.get('dropout_rate')}")
    print(f"Best learning_rate: {best_hps.get('learning_rate')}")
    print(f"Best batch_size: {best_hps.get('batch_size')}")
    # print(f"Best is_transfer: {best_hps.get('is_transfer')}")

    # 获取最佳模型
    best_model = tuner.get_best_models(num_models=1)[0]

    # 进行一次训练以确保 history 对象可用
    best_model_history = best_model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs
    )

    # 保存最佳模型
    best_model.save("models/best_model.h5")

    # 绘制训练过程曲线
    show_loss_acc(best_model_history)  # 使用 fit 返回的 history


# 主程序入口
if __name__ == '__main__':
    # 使用超参数调优训练模型
    train_with_tuner(epochs=15)
