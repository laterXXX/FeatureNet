import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dense, Flatten, add, BatchNormalization
from keras.models import Sequential
import data_handler
from keras.models import Model
from keras.optimizers import Adam
from data_handler import DataProducer

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# 输入数据为 mnist 数据集
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 　x_train = x_train / 255
# 　x_test = x_test / 255

# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)
class orbnet():

    def __init__(self, epochs, batchsize):
        self.le = self.res34()
        self.le.summary()
        self.optimizer = Adam(0.002, 0.5)

        self.epochs = epochs
        self.batchsize = batchsize
        self.le.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])
        self.path = 'F:/paper/101_ObjectCategories/101_ObjectCategories/'
        self.dh = DataProducer(self.path)

    def res34(self):
        def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
            if name is not None:
                bn_name = name + '_bn'
                conv_name = name + '_conv'
            else:
                bn_name = None
                conv_name = None

            x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
            x = BatchNormalization(axis=3, name=bn_name)(x)
            return x

        def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
            x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
            x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
            if with_conv_shortcut:
                shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
                x = add([x, shortcut])
                return x
            else:
                x = add([x, inpt])
                return x

        inpt = Input(shape=(128, 128, 1))

        x = Conv2d_BN(inpt, nb_filter=64, kernel_size=(3, 3), strides=1, padding='same')

        x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))

        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=1, with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))

        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=1, with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))

        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=1, with_conv_shortcut=True)
        x = Conv2D(8, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)

        model = Model(inputs=inpt, outputs=x)
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        return model

    def lenet(self):
        input = Input(shape=(128, 128, 1))
        # 选取6个特征卷积核，大小为5∗5(不包含偏置),得到66个特征图，每个特征图的大小为32−5+1=2832−5+1=28，
        # 也就是神经元的个数由10241024减小到了28∗28=78428∗28=784。
        # 输入层与C1层之间的参数:6∗(5∗5+1)
        c1 = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input)
        # 这一层的输入为第一层的输出，是一个28*28*6的节点矩阵。
        # 本层采用的过滤器大小为2*2，长和宽的步长均为2，所以本层的输出矩阵大小为14*14*6。
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # 本层的输入矩阵大小为14*14*6，使用的过滤器大小为5*5，深度为16.本层不使用全0填充，步长为1。
        # 本层的输出矩阵大小为10*10*16。本层有5*5*6*16+16=2416个参数
        c2 = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c1)

        c3 = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c2)

        c4 = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c3)
        c5 = Conv2D(8, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c4)
        # model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        # model.summary()
        return Model(input, c5)

    def train(self, ):
        t = 0
        for epoch in range(self.epochs):
            imgs, label = self.dh.get_data(self.batchsize)
            loss = self.le.train_on_batch(imgs, label)
            print("trainloss----:", loss)
            t += 1
            if t % 10 == 0 and t > 0:
                valid_imgs, valid_label = self.dh.get_data(self.batchsize + 16, 'valid')
                valid_loss = self.le.evaluate(valid_imgs, valid_label)
                print("valid_loss----:", valid_loss)

if __name__ == "__main__":
    epochs = 2111
    batchsize = 8
    orbnet = orbnet(epochs, batchsize)
    orbnet.train()

# model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test)
# print('Test Loss:', score[0])
# print('Test accuracy:', score[1])
