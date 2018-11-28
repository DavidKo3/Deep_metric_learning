from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16 , preprocess_input

from keras.layers.convolutional import AtrousConvolution2D
from keras.layers import Input, Dense, GlobalMaxPooling2D, Dropout, Lambda , GlobalAveragePooling2D
import numpy as np
import os
from keras.models import load_model
from keras import backend as K
from keras.models import Model

import umap
import pylab as pl
import glob
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import time
import pickle
import cv2


import re
from random import sample
import scipy.spatial.distance as dist

from sklearn.preprocessing import normalize

model_resnet50 = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model_vgg16 = VGG16(weights='imagenet', include_top=False, pooling='avg')
embedding_dim = 128 # 5


# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
def GetBaseModel(base_model):
    # base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    # x = GlobalMaxPooling2D()(x)
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.5)(x)
    # dense_1 = Dense(embedding_dim)(x)
    dense_1 = Dense(embedding_dim, activation='relu')(x)
    normalized = Lambda(lambda x: K.l2_normalize(x, axis=1))(dense_1)
    embedding = Model(base_model.input, normalized, name="base_model")

    return embedding

    # normalized = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
    # dense_1 = Dense(embedding_dim,activation='relu')(normalized)
    # base_model = Model(base_model.input, dense_1, name="base_model")
    #
    # return base_model

embedding_model = GetBaseModel(base_model)

def feature_ext(img_path):
    # img_path ="./deal_img/deal3.jpg"

    img = image.load_img(img_path, target_size=(224, 224 )) #(224, 224)
    img_width, img_height = img.size
    img_data = image.img_to_array(img) # (224, 224, 3)
    img_data = np.expand_dims(img_data, axis=0) # (1, 224, 224, 3)
    img_data = preprocess_input(img_data)

    embedding = embedding_model.predict(img_data)
    # print("embedding.shape:", embedding.shape)
    return embedding
    # res50_feature = model_resnet50.predict(img_data)
    #
    #
    # res50_feature = normalize(res50_feature)
    #
    # return res50_feature


def list_pic(dir, ext='png|jpg|jepg|bmp|'):
    return [os.path.join(root, f) for root, _, files in os.walk(dir) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f)]


root = "./deep_fashion_class_20_100"
triplet_data_dir = "./triplet_sample_list/"
def get_pos_img(img_name, img_names, num_pos_img):
    random_num = np.arange(len(img_names))
    np.random.shuffle(random_num)
    if int(num_pos_img) > (len(img_names) - 1):
        num_pos_img = len(img_names) - 1
    pos_cnt = 0
    pos_img = list()
    for rnd_ind in list(random_num):
        if img_names[rnd_ind] != img_name :
            pos_img.append(img_names[rnd_ind])
            pos_cnt += 1
            if int(pos_cnt) > (int(num_pos_img) - 1):
                break

    return pos_img


def get_neg_img(all_imgs, img_names, num_neg_img):
    random_num = np.arange(len(all_imgs))
    np.random.shuffle(random_num)
    if int(num_neg_img) > (len(all_imgs)-1):
        random_num = len(all_imgs) - 1
    neg_cnt = 0
    neg_img = list()
    for rnd_ind in list(random_num):
        if all_imgs[rnd_ind] not in img_names: # if each image is not in a particluar class, it is treat as negative for that class
            neg_img.append(all_imgs[rnd_ind])
            neg_cnt += 1
            if int(neg_cnt) > (int(num_neg_img) - 1):
                break

    return neg_img

def distance(a, b):
    return (np.square(a - b)).sum()

# def pair_dis

def get_neg_img_sampling(all_imgs, img_names, num_neg_img):
    """
    :param all_imgs:
    :param img_names:
    :param num_neg_img:
    :return:
    """
    random_num = np.arange(len(all_imgs))
    np.random.shuffle(random_num)
    if int(num_neg_img) > (len(all_imgs)-1):
        random_num = len(all_imgs) - 1
    neg_cnt = 0
    neg_img = list()
    rnd_img_feature_dict= dict()
    for rnd_ind in list(random_num):
        rnd_img_feature_dict[rnd_ind] = feature_ext(all_imgs[rnd_ind])

    some_img_feature_dict = dict()
    some_class_num = np.arange(len(img_names))
    for sc_ind in list(some_class_num):
        some_img_feature_dict[img_names[sc_ind]] = feature_ext(img_names[sc_ind])

    for rnd_ind in list(random_num):
        if all_imgs[rnd_ind] not in img_names: # if each image is not in a particular class, it is treat as negative for that class
            neg_img.append(all_imgs[rnd_ind])
            neg_cnt += 1
            if int(neg_cnt) > (int(num_neg_img) - 1):
                break

    return neg_img

# def triplet_sampler(dir_path, output_path, num_neg_img, num_pos_img):
#     classes = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
#     all_img = []
#     # print("classes : ", classes)
#     for class_ in classes:
#         all_img += (list_pic(os.path.join(dir_path, class_)))
#
#     triplet_lists = list()
#     for class_ in classes:
#         # image list in a class
#         img_names = list_pic(os.path.join(dir_path, class_))
#         for img_name in img_names: # each image in some class
#             img_name_set = set(img_names)
#             query_img = img_name
#             # sampling positive image
#             pos_img_list = get_pos_img(img_name, img_names, num_pos_img)
#             for pos_img in pos_img_list:
#                 neg_img_list = get_neg_img(all_img, set(img_names), num_neg_img)
#                 for neg_img in neg_img_list:
#                     triplet_lists.append(query_img+',')
#                     triplet_lists.append(pos_img + ',')
#                     triplet_lists.append(neg_img + '\n')
#
#     with open(os.path.join(output_path, "triplet_list.txt"), mode='w', encoding='UTF-8') as f:
#         f.write("".join(triplet_lists))

def triplet_sampler(dir_path, output_path, num_neg_img, num_pos_img):
    classes = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    all_img = []
    # print("classes : ", classes)
    for class_ in classes:
        all_img += (list_pic(os.path.join(dir_path, class_)))

    triplet_lists = list()
    test_triplet_lists = list()
    for class_ in classes:
        # image list in a class
        img_names = list_pic(os.path.join(dir_path, class_))
        for img_name in img_names: # each image in some class
            img_name_set = set(img_names)
            query_img = img_name
            # sampling positive image
            pos_img_list = get_pos_img(img_name, img_names, num_pos_img)
            for pos_img in pos_img_list:
                neg_img_list = get_neg_img(all_img, set(img_names), num_neg_img)

                for neg_img in neg_img_list:
                    # calculation of pos , neg dist
                    q_pos_dist = distance(feature_ext(query_img), feature_ext(pos_img))
                    q_neg_dist = distance(feature_ext(query_img), feature_ext(neg_img))

                    # queue_neg_img_list = neg_img_list
                    # while ( q_pos_dist >= q_neg_dist and len(queue_neg_img_list)>0 ):
                    #     len_neg_img_list = len(queue_neg_img_list)
                    #
                    #     neg_img = queue_neg_img_list.pop(0)
                    #     q_neg_dist = distance(feature_ext(query_img), feature_ext(neg_img))

                    # q_neg_dist = distance(feature_ext(query_img), feature_ext(neg_img))
                    # print("=================================")
                    # print("q_pos_dist : ", q_pos_dist)
                    # print("q_neg_dist : ", q_neg_dist)
                    # print("=================================")

                    # if ( (q_pos_dist - q_neg_dist< 0) and (q_pos_dist - q_neg_dist + 0.3 > 0 )):
                    if ((q_pos_dist - q_neg_dist < 0) ):
                        print("-"*20)
                        print("q_pos_dist - q_neg_dist : ", q_pos_dist - q_neg_dist)
                        print("q_pos_dist :", distance(feature_ext(query_img), feature_ext(pos_img)))
                        print("q_pos_dist :", q_pos_dist)
                        print("q_neg_dist :", distance(feature_ext(query_img), feature_ext(neg_img)))
                        print("q_neg_dist :", q_neg_dist)

                        print("-" * 20)
                        triplet_lists.append(query_img + ',')
                        triplet_lists.append(pos_img + ',')
                        triplet_lists.append(neg_img + '\n')
                    if (q_pos_dist - q_neg_dist> 0):
                        # print("/"*20)
                        # print("q_pos_dist - q_neg_dist : ", q_pos_dist - q_neg_dist)
                        # print("q_pos_dist :", distance(feature_ext(query_img), feature_ext(pos_img)))
                        # print("q_pos_dist :", q_pos_dist)
                        # print("q_neg_dist :", distance(feature_ext(query_img), feature_ext(neg_img)))
                        # print("q_neg_dist :", q_neg_dist)
                        # print("/" * 20)
                        test_triplet_lists.append(query_img + ',')
                        test_triplet_lists.append(pos_img + ',')
                        test_triplet_lists.append(neg_img + '\n')

    with open(os.path.join(output_path, "triplet_list_semi_hard_sampling.txt"), mode='w+', encoding='UTF-8') as f,\
        open(os.path.join(output_path, "triplet_for_test.txt"), mode='w+', encoding='UTF-8') as test_tri:
        f.write("".join(triplet_lists))
        test_tri.write("".join(test_triplet_lists))





triplet_sampler("./deep_fashion_class_5_100", triplet_data_dir, 5, 2)

total_dir = "./triplet_sample_list/triplet_list_semi_hard_sampling.txt"
valid_dir = "./triplet_sample_list/triplet_valid_list_semi_hard_sampling.txt"
test_dir = "./triplet_sample_list/triplet_test_list_semi_hard_sampling.txt"
train_dir = "./triplet_sample_list/triplet_train_list_semi_hard_sampling.txt"
def sep_dataset(total_dir, valid_dir, test_dir, train_dir):
    with open(total_dir, "rt") as f, \
        open(valid_dir, "w+", encoding='UTF-8') as w_valid, \
        open(test_dir, "w+", encoding='UTF-8') as w_test, \
        open(train_dir, "w+", encoding='UTF-8') as w_train:
            tolist = [line for line in f]
            num_total_list = len(tolist)
            portion_to_valid = int(round(num_total_list * 0.2))
            valid = sample(tolist, portion_to_valid)
            num_valid = len(valid)
            not_valid_list = [x for x in tolist if x not in valid]
            num_not_valid_list = len(not_valid_list)

            portion_to_test = int(round(num_total_list * 0.2))
            test = sample(not_valid_list, portion_to_test)

            num_test = len(test)

            train = [x for x in not_valid_list if x not in test]
            w_valid.write("".join(valid))
            w_test.write("".join(test))
            w_train.write("".join(train))

    print("completed separation of datatset !!")


sep_dataset(total_dir, valid_dir, test_dir, train_dir)






















