# -*-coding:utf-8-*-
"""
Super-resolution of CelebA using Generative Adversarial Networks.

The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0

Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to 'datasets/'
4. Run the sript using command 'python srgan.py'
"""

from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, MaxPool2D, AvgPool2D, MaxPooling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.applications import VGG19, inception_resnet_v2, inception_v3, VGG16, ResNet50
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
import datetime
import matplotlib.pyplot as plt
import sys
import data_handler as DataLoader
import numpy as np
import os
import keras as K


class FeatureNet():
    def __init__(self):
        # Input shape
        self.channels = 3
        self.lr_height = 128  # Low resolution height
        self.lr_width = 128  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = 128  # High resolution height
        self.hr_width = 128  # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = 8

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # optimizer = Adam(0.0002, 0.5)
        # optimizer1 = RMSprop(lr=0.0001)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # Configure data loader
        self.dataset_name = 'random_dataset'
        self.predict_dir = 'predict'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_width, self.hr_height))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (2, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        # self.discriminator = self.build_discriminator()
        # self.discriminator.summary()
        # self.discriminator.compile(loss='mse',
        #     optimizer=optimizer,
        #     metrics=['accuracy'])

        # Build and compile the critic
        self.discriminator = self.build_critic()
        self.discriminator.compile(loss=self.wasserstein_loss,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.dense_gener()
        self.generator.summary()
        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        self.combined = Model([img_lr, img_hr], [validity, fake_features])

        self.combined.compile(loss=[self.wasserstein_loss, 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)

    def lenet(self):
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c0 = Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3))

        c1 = Conv2D(16, kernel_size=(3, 3), activation='relu')(c0)

        c2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(c1)

        c3 = Conv2D(16, kernel_size=(3, 3), activation='relu')(c2)

        c4 = Conv2D(16, kernel_size=(3, 3), activation='relu')(c3)

        c5 = Conv2D(3, kernel_size=(3, 3), activation='relu')(c4)
        return Model(img_lr, c5)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.hr_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.hr_shape)
        validity = model(img)

        return Model(img, validity)

    def build_vgg(self):

        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        #
        # vgg = VGG16()
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        # vgg.outputs = [vgg.layers[9].output]

        # img = Input(shape=self.hr_shape)

        # Extract image features
        #
        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.hr_shape)
        img_features = vgg(img)

        # res = ResNet50()
        # img = Input(shape=self.hr_shape)
        # img_features = res(img)

        return Model(img, img_features)

    def dense_gener(self):
        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            # d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(d)
            d = Activation('relu')(d)
            # d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        fil = 1
        # Low resolution image input

        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c0 = Conv2D(fil, kernel_size=3, strides=1, padding='same')(img_lr)
        c0 = LeakyReLU(alpha=0.2)(c0)

        # e0 = Conv2D(1, kernel_size=3, strides=1, padding='same')(c0)
        # e0 = BatchNormalization(momentum=0.8)(e0)
        # e0 = LeakyReLU(alpha=0.2)(e0)
        rr1 = residual_block(c0, fil)
        for _ in range(self.n_residual_blocks - 1):
            rr1 = residual_block(rr1, fil)

        rr2 = Add()([c0, rr1])
        rr2 = residual_block(rr2, fil)
        for _ in range(self.n_residual_blocks - 1):
            rr2 = residual_block(rr2, fil)

        rr3 = Add()([rr1, rr2])
        rr3 = residual_block(rr3, fil)
        for _ in range(self.n_residual_blocks - 1):
            rr3 = residual_block(rr3, fil)

        # l0 = Conv2D(16, kernel_size=9, strides=1, padding='same')(img_lr)
        # l0 = LeakyReLU(alpha=0.2)(l0)
        #
        # e1 = Conv2D(16, kernel_size=3, strides=1, padding='same')(l0)
        # e1 = BatchNormalization(momentum=0.8)(e1)
        # e1 = LeakyReLU(alpha=0.2)(e1)
        #
        # rr2 = residual_block(e1, fil)
        # for _ in range(self.n_residual_blocks - 1):
        #     rr2 = residual_block(rr2, fil)
        #
        # ll0 = Conv2D(16, kernel_size=27, strides=1, padding='same')(img_lr)
        # ll0 = LeakyReLU(alpha=0.2)(ll0)
        #
        # ee1 = Conv2D(16, kernel_size=3, strides=1, padding='same')(ll0)
        # ee1 = BatchNormalization(momentum=0.8)(ee1)
        # ee1 = LeakyReLU(alpha=0.2)(ee1)

        # rr3 = residual_block(ee1, fil)
        # for _ in range(self.n_residual_blocks - 1):
        #     rr3 = residual_block(rr3, fil)

        dens0 = Concatenate()([rr3, rr2, rr1, c0, img_lr])

        # for _ in range(self.n_residual_blocks - 1):
        #     e0 = residual_block(e0, self.gf)

        # fusion
        f0 = Conv2D(64, kernel_size=(3, self.lr_width), strides=1, padding="same")(dens0)
        f0 = Activation('relu')(f0)
        # f0 = BatchNormalization(momentum=0.8)(f0)
        # f0 = Reshape((128,128,64))(f0)
        # r = residual_block(f0, self.gf)
        # for _ in range(self.n_residual_blocks - 1):
        #     r = residual_block(r, self.gf)
        # f1 = Add()([f0,r])

        # c1 = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same')(f0)
        # c1 = Activation('relu')(c1)
        # c1 = BatchNormalization(momentum=0.8)(c1)

        # Propogate through residual blocks
        # r = residual_block(c1, self.gf)
        # for _ in range(self.n_residual_blocks - 1):
        #     r = residual_block(r, self.gf)

        # ,,,,
        # Post-residual block
        # c2 = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same')(f0)
        # c2 = Activation('relu')(c2)
        # c2 = BatchNormalization(momentum=0.8)(c2)
        #
        # d0 = Conv2D(32, kernel_size=3, strides=1, padding="same")(c2)
        # d0 = Activation('relu')(d0)
        # d0 = BatchNormalization(momentum=0.8)(d0)
        # ,,,,
        # generate
        #
        # g2 = Conv2D(128, kernel_size=3, strides=1, padding="same")(d0)
        # g2 = BatchNormalization(momentum=0.8)(g2)
        # g2 = Activation('relu')(g2)

        #
        # g3 = Conv2D(128, kernel_size=3, strides=1, padding="same")(g2)
        # g3 = BatchNormalization(momentum=0.8)(g3)
        # g3 = Activation('relu')(g3)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(f0)

        return Model(img_lr, gen_hr)

    def bild_autoencoder(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c0 = Conv2D(32, kernel_size=9, strides=1, padding='same')(img_lr)
        c0 = BatchNormalization(momentum=0.8)(c0)
        c0 = LeakyReLU(alpha=0.2)(c0)

        e0 = Conv2D(32, kernel_size=(3, 5), strides=2, padding='same')(c0)
        e0 = BatchNormalization(momentum=0.8)(e0)
        e0 = LeakyReLU(alpha=0.2)(e0)
        #
        e1 = Conv2D(64, kernel_size=(3, 5), strides=2, padding='same')(e0)
        e1 = BatchNormalization(momentum=0.8)(e1)
        e1 = LeakyReLU(alpha=0.2)(e1)

        # for _ in range(self.n_residual_blocks - 1):
        #     e0 = residual_block(e0, self.gf)

        # fusion
        f0 = Conv2D(128, kernel_size=(1, self.lr_width), strides=1, padding="same")(e1)
        f0 = Activation('relu')(f0)
        f0 = BatchNormalization(momentum=0.8)(f0)

        # c1 = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same')(f0)
        # c1 = Activation('relu')(c1)
        # c1 = BatchNormalization(momentum=0.8)(c1)

        # Propogate through residual blocks
        # r = residual_block(c1, self.gf)
        # for _ in range(self.n_residual_blocks - 1):
        #     r = residual_block(r, self.gf)
        u = UpSampling2D(size=2)(f0)
        # Post-residual block
        c2 = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same')(u)
        c2 = Activation('relu')(c2)
        c2 = BatchNormalization(momentum=0.8)(c2)

        u1 = UpSampling2D(size=2)(c2)
        d0 = Conv2D(32, kernel_size=3, strides=1, padding="same")(u1)
        d0 = Activation('relu')(d0)
        d0 = BatchNormalization(momentum=0.8)(d0)

        # generate
        #
        # g2 = Conv2D(128, kernel_size=3, strides=1, padding="same")(d0)
        # g2 = BatchNormalization(momentum=0.8)(g2)
        # g2 = Activation('relu')(g2)

        #
        # g3 = Conv2D(128, kernel_size=3, strides=1, padding="same")(g2)
        # g3 = BatchNormalization(momentum=0.8)(g3)
        # g3 = Activation('relu')(g3)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(d0)

        return Model(img_lr, gen_hr)

    def bild_autoencoder_29_18_11(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            # d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c0 = Conv2D(64, kernel_size=3, strides=1, padding='same')(img_lr)
        c0 = Activation('relu')(c0)

        # d0 = Conv2D(256, kernel_size=3, strides=(1,2), padding="same")(c0)
        # d0 = Activation('relu')(d0)

        # d0 = Conv2D(64, kernel_size=3, strides=1, padding="same")(d0)
        # d0 = Activation('relu')(d0)
        # d0 = Conv2D(64, kernel_size=3, strides=1, padding="same")(d0)
        # d0 = Activation('relu')(d0)
        #
        # d1 = Conv2D(64, kernel_size=3, strides=1, padding="same")(d0)
        # d1 = Activation('relu')(d1)

        c1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(c0)
        c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = Conv2D(128, kernel_size=(1, 270), strides=(1, 2), padding='same')(r)
        c2 = Activation('relu')(c2)
        c2 = BatchNormalization(momentum=0.8)(c2)

        # c2 = Add()([c2, c1])

        # resize feature_map

        r1 = Conv2D(64, kernel_size=9, strides=1, padding="same")(c2)
        r1 = Activation('relu')(r1)

        # zip
        # z1 = Conv2D(256, kernel_size=9, strides=(1,2), padding="same")(c2)
        # z1 = Activation('relu')(z1)

        # z1 = Conv2D(64, kernel_size=3, strides=(1,2), padding="same")(z1)
        # z1 = Activation('relu')(z1)

        # unresize
        # u1 = UpSampling2D(size=(1,2))(z1)
        # u1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(u1)
        # u1 = Activation('relu')(u1)

        # generate
        # g0 = Conv2D(128, kernel_size=9, strides=1, padding="same")(c2)
        # g0 = Activation('relu')(g0)
        #
        # g1 = Conv2D(256, kernel_size=3, strides=1, padding="same")(g0)
        # g1 = Activation('relu')(g1)
        #
        #
        g2 = Conv2D(256, kernel_size=9, strides=1, padding="same")(r1)
        g2 = BatchNormalization(momentum=0.8)(g2)
        g2 = Activation('relu')(g2)
        g2 = Conv2D(256, kernel_size=3, strides=1, padding="same")(g2)
        g2 = BatchNormalization(momentum=0.8)(g2)
        g2 = Activation('relu')(g2)
        #
        g3 = Conv2D(128, kernel_size=3, strides=1, padding="same")(g2)
        g3 = BatchNormalization(momentum=0.8)(g3)
        g3 = Activation('relu')(g3)

        # Upsampling
        # u1 = deconv2d(c2)
        # u2 = deconv2d(u1)
        # u1 = Conv2D(256, kernel_size=3, strides=1, padding="same")(c2)
        # u1 = Activation('relu')(u1)
        #
        # u2 = Conv2D(256, kernel_size=3, strides=1, padding="same")(u1)
        # u2 = Activation('relu')(u2)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=1, strides=1, padding='same', activation='tanh')(g3)

        return Model(img_lr, gen_hr)

    def build_generator(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c0 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c0 = Activation('relu')(c0)

        d1 = Conv2D(64, kernel_size=(3, self.lr_width), strides=1, padding="same")(c0)
        d1 = Activation('relu')(d1)

        c1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(d1)
        c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = Activation('relu')(c2)
        c2 = BatchNormalization(momentum=0.8)(c2)

        c2 = Add()([c2, c1])

        # Upsampling
        # u1 = deconv2d(c2)
        # u2 = deconv2d(u1)
        u1 = Conv2D(256, kernel_size=3, strides=1, padding="same")(c2)
        u1 = Activation('relu')(u1)

        u2 = Conv2D(256, kernel_size=3, strides=1, padding="same")(u1)
        u2 = Activation('relu')(u2)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df * 2)
        d4 = d_block(d3, self.df * 2, strides=2)
        d5 = d_block(d4, self.df * 4)
        d6 = d_block(d5, self.df * 4, strides=2)
        d7 = d_block(d6, self.df * 8)
        d8 = d_block(d7, self.df * 8, strides=2)
        # d8 = Flatten()(d8)
        d9 = Dense(self.df * 16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='tanh')(d10)

        return Model(d0, validity)

    def predict(self):
        os.makedirs('images/%s' % self.predict_dir, exist_ok=True)
        r, c = 2, 2

        self.generator.load_weights('generator_5_6')

        # Sample images and their conditioning counterparts
        imgs_hr, imgs_lr = self.data_loader.predict_realdata(2)

        # From low res. image generate high res. version
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (self.predict_dir, 1))
        plt.close()

        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%s/%d_lowres%d.png' % (self.predict_dir, 1, i))
            plt.close()

    def train(self, epochs, batch_size=20, sample_interval=500):

        start_time = datetime.datetime.now()
        dloss = []
        gloss = []

        a = os.path.isfile('./generator_5_6')
        if a:
            self.generator.load_weights('generator_5_6')
            self.discriminator.load_weights('discriminator_5_6', True)
            # self.combined.load_weights('combined_5_6', True)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):
            for _ in range(self.n_critic):

                # ----------------------
                #  Train Discriminator
                # ----------------------

                # Sample images and their conditioning counterparts
                imgs_hr, imgs_lr = self.data_loader.load_data_random(batch_size)

                # From low res. image generate high res. version
                fake_hr = self.generator.predict(imgs_lr)

                # valid = np.ones((batch_size,) + self.disc_patch)
                # fake = np.zeros((batch_size,) + self.disc_patch)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
                d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Clip critic weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data_random(batch_size)

            # The generators want the discriminators to label the generated images as real
            # valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print("%d %s gloss--%s d_loss -- %s time" % (epoch, g_loss, d_loss, elapsed_time,))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            if epoch % 500 == 0 and epoch > 400:
                self.generator.save_weights('generator_5_6', True)
                self.discriminator.save_weights('discriminator_5_6', True)
                self.combined.save_weights('combined_5_6', True)

    def sample_images(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data_random(batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()

        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%s/%d_lowres%d.png' % (self.dataset_name, epoch, i))
            plt.close()


if __name__ == '__main__':
    gan = FeatureNet()
    gan.train(epochs=300000, batch_size=2, sample_interval=50)
    # gan.predict()
