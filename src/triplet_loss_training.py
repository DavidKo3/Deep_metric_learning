
from keras.regularizers import l2
from keras.optimizers import SGD

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16 , preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K
import numpy as np
import os


import numpy as np
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam , SGD
from keras.layers import Input, Dense, GlobalMaxPooling2D, Dropout, Lambda, GlobalAveragePooling2D




import cv2
import os
from keras.models import load_model
import keras

from keras.utils import np_utils
from scipy.misc import imresize
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# model_vgg16 = VGG16(weights='imagenet', include_top=False, pooling='avg')
#
# def feature_ext(img_data):
#     # img_path ="./deal_img/deal3.jpg"
#
#     # img = image.load_img(img_path, target_size=(32, 32))
#     # img_width, img_height = img.size
#     # img_data = image.img_to_array(img) # (32, 32, 3)
#     img_data = np.expand_dims(img_data, axis=0) # (1, 32, 32, 3)
#     img_data = preprocess_input(img_data)
#     vgg_16_feature = model_vgg16.predict(img_data)
#
#     return vgg_16_feature




############## Settings #################
batch_size = 30 # 8
image_size = 224 # 224
embedding_dim = 128 # 5
Margin_K = 0.5
def read_images(img_path):
    img = image.load_img(img_path, target_size=(image_size, image_size))
    img_data = image.img_to_array(img) # (200, 200, 3)
    return img_data


def get_imge_list(file_dir):
    files_list = list()
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            files_list.append(line.split())

    return files_list

DATA_DIR = "./triplet_sample_list"
IMAGE_DIR = os.path.join(DATA_DIR, "triplet_train_list_semi_hard_sampling.txt")
files_train_list = get_imge_list(IMAGE_DIR)

# print(files_train_list)
ref_image = plt.imread("".join(files_train_list[3]).split(",")[0])
sim_image = plt.imread("".join(files_train_list[3]).split(",")[1])
dif_image = plt.imread("".join(files_train_list[3]).split(",")[2])


def draw_image(subplot, image, title):
    plt.subplot(subplot)
    plt.imshow(image)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


draw_image(131, ref_image, "reference")
draw_image(132, sim_image, "similar")
draw_image(133, dif_image, "different")
plt.tight_layout()
plt.show()

# def image_generator_v2(files,  batch_size=3):
#
#     while True:
#         num_records = len(files)
#         indices = np.random.permutation(np.arange(num_records))
#         num_batches = num_records // batch_size
#
#         batch_input_anchor = []
#         batch_input_pos = []
#         batch_input_neg = []
#         batch_output = list()
#
#         for bid in range(num_batches):
#             batch_indices = indices[bid*batch_size:(bid+1)*batch_size]
#             batch = [files[i] for i in batch_indices]
#             # print("num_recodes : ", num_records)
#             # print("batch size :", batch_size)
#             # print("num_batches :", num_batches)
#             # print("batch_indices : ", batch_indices)
#             # print("files[bid] : ", files[bid])
#             # print("batch : ", batch)
#             batch_input_anchor = [read_images("".join(b).split(",")[0]) for b in batch]
#             batch_input_pos = [read_images("".join(b).split(",")[1]) for b in batch]
#             batch_input_neg = [read_images("".join(b).split(",")[2]) for b in batch]
#
#
#
#         anc = preprocess_input(np.array(batch_input_anchor, dtype='float32')) # (batch_size, 224, 224, 3)
#         pos = preprocess_input(np.array(batch_input_pos, dtype='float32'))    # (batch_size, 224, 224, 3)
#         neg = preprocess_input(np.array(batch_input_neg, dtype='float32'))    # (batch_size, 224, 224, 3)
#         label = np.ones(batch_size)
#
#         base_model = GetBaseModel()
#
#         # semi hard negative sampling
#         for bid in range(batch_size):
#             anc_d = feature_ext(anc[bid])
#             pos_d = feature_ext(pos[bid])
#             nec_d = feature_ext(neg[bid])
#             # anc_d = feature_ext(np.expand_dims(anc[bid], axis=0).shape)
#             # pos_d = feature_ext(np.expand_dims(pos[bid], axis=0).shape)
#             # nec_d = feature_ext(np.expand_dims(neg[bid], axis=0).shape)
#             d     = np.square(anc_d - pos_d).sum()
#             neg_d = np.square(anc_d - nec_d).sum()
#             # d = np.square(anc[bid] - pos[bid]).sum()
#             # neg_d = np.square(anc[bid] - neg[bid]).sum()
#             cnt = 0
#             while d > neg_d:
#                 # rnd_ind = random.sample(range(batch_size), 1)[0]
#                 rnd_ind = cnt
#                 print("rnd_ind :", rnd_ind)
#                 neg[bid] = neg[rnd_ind]
#                 neg_d = np.square(anc[bid] - neg[bid]).sum()
#                 cnt+=1
#                 if cnt >= batch_size:
#                     neg[bid] = neg[rnd_ind]
#                     break
#             if d> neg_d:
#                 print("d : ", d)
#                 print("neg_d :", neg_d)
#
#
#
#         yield [anc, pos, neg], label
datagen = ImageDataGenerator(rotation_range=45,
                               width_shift_range=0.05,
                                height_shift_range=0.05,
                                zoom_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=True,
                            )

def image_generator(files, batch_size=3, augmentation=False):

    while True:
        num_records = len(files)
        indices = np.random.permutation(np.arange(num_records))
        num_batches = num_records // batch_size

        batch_input_anchor = []
        batch_input_pos = []
        batch_input_neg = []
        batch_output = list()

        for bid in range(num_batches):
            batch_indices = indices[bid*batch_size:(bid+1)*batch_size]
            batch = [files[i] for i in batch_indices]
            # print("num_recodes : ", num_records)
            # print("batch size :", batch_size)
            # print("num_batches :", num_batches)
            # print("batch_indices : ", batch_indices)
            # print("files[bid] : ", files[bid])
            # print("batch : ", batch)
            batch_input_anchor = [read_images("".join(b).split(",")[0]) for b in batch]
            batch_input_pos = [read_images("".join(b).split(",")[1]) for b in batch]
            batch_input_neg = [read_images("".join(b).split(",")[2]) for b in batch]



        anc = preprocess_input(np.array(batch_input_anchor, dtype='float32')) # (batch_size, 224, 224, 3)
        pos = preprocess_input(np.array(batch_input_pos, dtype='float32'))    # (batch_size, 224, 224, 3)
        neg = preprocess_input(np.array(batch_input_neg, dtype='float32'))    # (batch_size, 224, 224, 3)
        label = np.ones(batch_size)


        if augmentation:
            # image data augmentation
            gen_anc = datagen.flow(anc, label, batch_size=batch_size, shuffle=False)
            gen_pos = datagen.flow(pos, label, batch_size=batch_size, shuffle=False)
            gen_neg = datagen.flow(neg, label, batch_size=batch_size, shuffle=False)

            anc_x1 = gen_anc.next()
            pos_x1 = gen_pos.next()
            neg_x1 = gen_neg.next()

            yield [anc_x1[0], pos_x1[0], neg_x1[0]], label
        else:
            yield [anc, pos, neg], label


def image_generator_no_augmentation(files, batch_size=3):

    while True:
        num_records = len(files)
        indices = np.random.permutation(np.arange(num_records))
        num_batches = num_records // batch_size

        batch_input_anchor = []
        batch_input_pos = []
        batch_input_neg = []
        batch_output = list()

        for bid in range(num_batches):
            batch_indices = indices[bid*batch_size:(bid+1)*batch_size]
            batch = [files[i] for i in batch_indices]
            # print("num_recodes : ", num_records)
            # print("batch size :", batch_size)
            # print("num_batches :", num_batches)
            # print("batch_indices : ", batch_indices)
            # print("files[bid] : ", files[bid])
            # print("batch : ", batch)
            batch_input_anchor = [read_images("".join(b).split(",")[0]) for b in batch]
            batch_input_pos = [read_images("".join(b).split(",")[1]) for b in batch]
            batch_input_neg = [read_images("".join(b).split(",")[2]) for b in batch]



        anc = preprocess_input(np.array(batch_input_anchor, dtype='float32')) # (batch_size, 224, 224, 3)
        pos = preprocess_input(np.array(batch_input_pos, dtype='float32'))    # (batch_size, 224, 224, 3)
        neg = preprocess_input(np.array(batch_input_neg, dtype='float32'))    # (batch_size, 224, 224, 3)
        label = np.ones(batch_size)

        yield [anc, pos, neg], label


def image_generator_for_test(files,  batch_size=3):

    while True:
        num_records = len(files)
        indices = np.random.permutation(np.arange(num_records))
        num_batches = num_records // batch_size

        batch_input_anchor = []
        batch_input_pos = []
        batch_input_neg = []
        batch_output = list()

        for bid in range(num_batches):
            batch_indices = indices[bid*batch_size:(bid+1)*batch_size]
            batch = [files[i] for i in batch_indices]

            batch_input_anchor = [read_images("".join(b).split(",")[0]) for b in batch]
            batch_input_pos = [read_images("".join(b).split(",")[1]) for b in batch]
            batch_input_neg = [read_images("".join(b).split(",")[2]) for b in batch]



        anc = preprocess_input(np.array(batch_input_anchor, dtype='float32')) # (batch_size, 224, 224, 3)
        pos = preprocess_input(np.array(batch_input_pos, dtype='float32'))    # (batch_size, 224, 224, 3)
        neg = preprocess_input(np.array(batch_input_neg, dtype='float32'))    # (batch_size, 224, 224, 3)
        label = np.ones(batch_size)

        yield [anc, pos, neg], label


# class DataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, labels="label", batch_size=8, dim=(224,224), n_channels=3,
#                  n_classes=10, shuffle=True):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()
#
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))
#
#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#
#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]
#
#         # Generate data
#         X = self.__data_generation(list_IDs_temp)
#
#         return X
#
#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
#
#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         batch_input_anchor = np.empty((self.batch_size, *self.dim, self.n_channels))
#         batch_input_pos = np.empty((self.batch_size, *self.dim, self.n_channels))
#         batch_input_neg = np.empty((self.batch_size, *self.dim, self.n_channels))
#         # y = np.empty((self.batch_size), dtype=int)
#         # batch_input_anchor = []
#         # batch_input_pos = []
#         # batch_input_neg = []
#
#         # Generate data
#         print("list_IDs_temp : ", list_IDs_temp)
#         for i, line in enumerate(list_IDs_temp):
#             # Store sample
#             batch_input_anchor[i,] = read_images("".join(list_IDs_temp[0]).split(",")[0])
#             batch_input_pos[i,] = read_images("".join(list_IDs_temp[0]).split(",")[1])
#             batch_input_neg[i,] = read_images("".join(list_IDs_temp[0]).split(",")[2])
#
#
#         label = np.ones(len(list_IDs_temp))
#             # Store class
#             # y[i] = self.labels[ID]
#         print("batch_input_anchor.shape ", batch_input_anchor.shape)
#         print("batch_input_pos.shape ", batch_input_pos.shape)
#         print("batch_input_neg.shape ", batch_input_neg.shape)
#         # return [batch_input_anchor], [batch_input_pos], [batch_input_neg]
#         yield [batch_input_anchor, batch_input_pos, batch_input_neg] , label

files_train_dir = "./triplet_sample_list/triplet_train_list_semi_hard_sampling.txt"

# files_test_dir = "./triplet_sample_list/triplet_test_list_semi_hard_sampling.txt"
files_test_dir = "./triplet_sample_list/triplet_for_test.txt"
files_valid_dir = "./triplet_sample_list/triplet_valid_list_semi_hard_sampling.txt"

files_train_list = get_imge_list(files_train_dir)
files_test_list = get_imge_list(files_test_dir)
files_valid_list = get_imge_list(files_valid_dir)



training_generator = image_generator(files_train_list, batch_size=batch_size, augmentation=False)
test_generator = image_generator(files_test_list, batch_size=batch_size, augmentation=False)
validation_generator = image_generator(files_valid_list, batch_size=batch_size, augmentation=False)




############## LOSS ###########################

def triplet_loss(y_true, y_pred):
    margin = K.constant(Margin_K)
    # return K.mean(K.maximum(K.constant(0.0), K.square(y_pred[:, 0, 0]) - K.square(y_pred[:, 1, 0]) + margin))
    return 0.5 * K.mean(K.maximum(K.constant(0.0), K.square(y_pred[:, 0, 0]) - K.square(y_pred[:, 1, 0]) + margin))
    # return K.mean(K.maximum(K.constant(0.0), K.square(y_pred[:, 0]) - K.square(y_pred[:, 1]) + margin))


def accuracy(y_true, y_pred):
    # return K.mean(y_pred[:, 0] < y_pred[:, 1])
    return K.mean(y_pred[:, 0, 0] < y_pred[:, 1, 0])
    # return K.mean(y_pred[:, 0, 0] - y_pred[:, 1, 0] + K.constant(0.4)> 0)

def euclidean_distance(vects):
    x, y = vects
    # return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def mean_pos_dist(_, y_pred):
    # return K.mean(y_pred[:, 0])
    return K.mean(y_pred[:, 0, 0])

def mean_neg_dist(_, y_pred):
    # return K.mean(y_pred[:, 1])
    return K.mean(y_pred[:, 1, 0])

############## Model ###########################

def GetBaseModel():
    # base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    # x = GlobalMaxPooling2D()(x)
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.5)(x)
    # dense_1 = Dense(embedding_dim, activation='relu')(x)
    # normalized = Lambda(lambda x: K.l2_normalize(x, axis=1))(dense_1)
    # base_model = Model(base_model.input, normalized, name="base_model")

    normalized = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
    dense_1 = Dense(embedding_dim, activation='relu')(normalized)
    base_model = Model(base_model.input, dense_1, name="base_model")

    return base_model


def GetMyModel(base_model):
    input_1 = Input((image_size, image_size, 3))
    input_2 = Input((image_size, image_size, 3))
    input_3 = Input((image_size, image_size, 3))

    net_anchor = base_model(input_1)
    net_positive = base_model(input_2)
    net_negative = base_model(input_3)

    positive_dist = Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
    negative_dist = Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])


    stacked_dists = Lambda(
        lambda vects: K.stack(vects, axis=1),
        name='stacked_dists'
    )([positive_dist, negative_dist])

    inputs = [input_1, input_2, input_3]


    model = Model(inputs=inputs, outputs=stacked_dists)

    # model.compile(loss=triplet_loss ,metrics=[accuracy], optimizer=Adam(0.000003))
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss={'stacked_dists': triplet_loss},
                  metrics={'stacked_dists': [accuracy, mean_pos_dist, mean_neg_dist]}, optimizer=sgd)
    # model.compile(loss={'stacked_dists' : triplet_loss}, metrics={'stacked_dists': [accuracy, mean_pos_dist, mean_neg_dist]}, optimizer=Adam(0.000003))
    # model.compile(loss={'stacked_dists': triplet_loss}, metrics={'stacked_dists': accuracy}, optimizer=Adam(0.0003))
    # model.compile(loss={'stacked_dists': triplet_loss},  optimizer=Adam(0.0001))


    return model

batch_test = next(test_generator)
base_model = GetBaseModel()
model = GetMyModel(base_model)
# #
# # #
# # # print(model.summary())
# # #
# print(np.array(batch_test[0]).shape) # (3, batch_size, 224, 224, 3)
# print(model.predict_on_batch(batch_test[0]))

checkpoint = ModelCheckpoint('./models/model_vgg16_triple_loss_v1.h5')
model.fit_generator(training_generator, validation_data=validation_generator, epochs=200, verbose=2, workers=4, steps_per_epoch=3, validation_steps=3, use_multiprocessing=False, callbacks=[checkpoint])

# trained_model = load_model('./models/model_resnet50_triple_loss.h5',  custom_objects={'triplet_loss':triplet_loss})
# trained_model = load_model('./models/model_resnet50_triple_loss.h5',  custom_objects={'triplet_loss':triplet_loss ,'accuracy':accuracy })
# trained_model = load_model('./models/model_vgg16_triple_loss_v1.h5', custom_objects={'triplet_loss':triplet_loss,'accuracy':accuracy, 'mean_pos_dist':mean_pos_dist, 'mean_neg_dist':mean_neg_dist})
trained_model = load_model('./models/model_vgg16_triple_loss_v1.h5', custom_objects={'triplet_loss':triplet_loss,'accuracy':accuracy, 'mean_pos_dist':mean_pos_dist, 'mean_neg_dist':mean_neg_dist})

result = trained_model.predict_on_batch(batch_test[0])

print(result)


#
# for i in range(5):
#     batch_test = next(test_generator)
#     # print(batch_test[0])
#     print("-" * 20)
#     print(model.predict_on_batch(batch_test[0]))
#     print("-"*20)
#     print("*" * 20)
#     print(trained_model.predict_on_batch(batch_test[0]))
#     print("*" * 20)