import os
import random
import cv2
import numpy as np

# read images --> generate sift\surf\orb features --> store the features information as a txt
class DataProducer:
    NUMBER_OF_TRAINING_SET = 4999
    NUMBER_OF_TESTING_SET = 2144
    NUMBER_OF_VALIDATION_SET = 1999

    NUMBER_OF_TRAIN_ITERATIONS = 0
    DATA_LIST = []

    def __init__(self, dataset_file='', img_res=(128, 128)):
        self.dataset_name = dataset_file
        self.img_res = img_res
        self.img_res_lr = (self.img_res[0] * 2, self.img_res[1])
        self.img_seize_lr = (self.img_res[0], self.img_res[1])
        self.ori__l = '/home/later/stitchingGAN/27/2019-04-27/1/58/*'
        self.ori__r = '/home/later/stitchingGAN/27/2019-04-27/1/59/*'



    def list_2_file(self, data, path):
        """
        :param data: list
        :param path: file path
        :return:
        """
        f = open(path, 'w+')
        for line in data:
            f.write(line + '\n')

    def get_all_image_path(self, images_folder):
        """
        :param images_folder: PATH: folder --> category --> images
        :return: A list include every image path
        """
        if not os.path.isdir(images_folder):
            return " can't open folder "

        folders = os.listdir(images_folder)
        images_path = []
        for folder in folders:
            if os.path.isdir(images_folder + folder):
                category = os.listdir(images_folder + folder)
                for image in category:
                    # print(images_folder + images_folder + image)
                    images_path.append(folder + '/' + image)
                    # images_path.append()
        # f = open(images_folder + 'images_path.txt', 'w+')
        if not os.path.exists(images_folder + 'images_path.txt'):
            self.list_2_file(images_path, images_folder + 'images_path.txt')
        return images_path

    def make_labels(self, file_path, file_name):
        """
        :param images_folder:
        :return:
        """
        if not os.path.exists(file_path + file_name):
            return 'path wrong'
        orb = cv2.ORB_create()
        arr = np.zeros((8, 128, 128))
        with open(file_path + file_name, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip('\n')
                category = line[:line.rindex('/')]

                if line[line.rfind('.') + 1:] == 'jpg':
                    img_name = line[line.rindex('/') + 1:line.rindex('.')].strip('\n')

                # with open( ,'w+') as txt_file:
                img = cv2.imread((file_path + line).strip('\n'), 0)

                img = cv2.resize(img, (128, 128), img)

                kps = orb.detect(img, None)
                del img

                for kp in kps:
                    angle = kp.angle
                    class_id = kp.class_id
                    octave = kp.octave
                    pt = kp.pt
                    response = kp.response
                    size = kp.size

                    x = int(pt[0])
                    y = int(pt[1])

                    arr[0, x, y] = 1
                    arr[1, x, y] = angle
                    arr[2, x, y] = class_id
                    arr[3, x, y] = octave
                    arr[4, x, y] = x
                    arr[5, x, y] = y
                    arr[6, x, y] = response
                    arr[7, x, y] = size

                np.save(file_path + category + '/' + img_name + '.npy', arr)
                arr[arr != 0] = 0




    def partition_data(self, images_folder):
        # 5000 train data 2000 validation data 2145 test data
        images_path = self.get_all_image_path(images_folder)

        train_set = []
        validation_set = []
        test_set = []
        for index, path in enumerate(images_path):
            if index < self.NUMBER_OF_TRAINING_SET:
                train_set.append(path)
            elif index < self.NUMBER_OF_VALIDATION_SET + self.NUMBER_OF_TRAINING_SET:
                validation_set.append(path)
            else:
                test_set.append(path)
        # print(len(train_set))
        # print(len(validation_set))
        # print(len(test_set))
        # test_file = open(images_folder + 'testing_set.txt', 'w+')
        self.list_2_file(test_set, images_folder + 'testing_set.txt')
        self.list_2_file(train_set, images_folder + 'train_set.txt')
        self.list_2_file(validation_set, images_folder + 'validation_set.txt')

    def get_img(self, data_set_path, stage='train', batch_size=64):

        # valid the path
        # return path in globals()
        if not os.path.isfile(data_set_path):
            return " can't open file"

        f = open(data_set_path, 'r')

        # 128 x 128 x 3byte = 49512 byte / 1024 = 48 M  per image
        # 48 M x 256 = 12 G
        # 48 M x 64 = 3 G
        data_set = []
        for i in batch_size:
            img = cv2.imread(f[i + self.NUMBER_OF_TRAIN_ITERATIONS])
            data_set.append(cv2.resize(img, (128,128)))

        data_set = np.array(data_set) / 127.5 - 1.

        # path_prefix = images_folder[:images_folder.rindex('/')] + '/'
        # print(path_prefix)

        images_num = 0  # total image 9145

        # 5000 train images, image size 128 x 128
        # read the image name first, then random read 1000 once

        # random
        # categories = len(folders)
        # random_categories = random.randint(0, categories)
        # if random_categories >= categories:
        #     random_categories = categories - 1
        # if random_categories < 0:
        #     random_categories = 0
        #
        # # category name
        # category_name = folders[categories]

        # print(random_categories)

        # for folder in folders:
        #     # print(folder)
        #     images = os.listdir(images_folder + folder)
        #     images_num += len(images)
        #     for image in images:
        #         # print(images_folder + image)
        #         img_path = images_folder + folder + '/' + image
        #         img = cv2.imread(img_path)
        #         # cv2.imshow('1', img)
        #         # cv2.waitKey(0)
        # print('images_num', images_num)
        return data_set

    def read_img(self, images_folder):
        pass

    def get_label(self):
        pass

    def get_data(self, batchsize):
        pass


if __name__ == '__main__':
    path = 'F:/paper/101_ObjectCategories/101_ObjectCategories/'
    data_producer = DataProducer(path)

    res = data_producer.make_labels(path, 'images_path.txt')
