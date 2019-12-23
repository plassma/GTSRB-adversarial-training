import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.image import array_to_img

from preprocessing import get_dataset, signnames

sys.path.append(os.getcwd())

LAMBDA = 0.2
RESULT_ROOT = "results"
TRAIN_PATH = "res\\train\\Final_Training\\Images\\"
TEST_PATH = "res\\test\\Final_Test\\Images\\"
TEST_LABELS_PATH = "res\\test\\Final_Test\\GT-final_test.csv"
IMG_SIZE = 64

# MIT-license: https://github.com/MaximilianIdahl/gtsrb-models-keras


def plot_labeled_images(rows, cols, images, labels):
    title_font_size = 8

    fig, axs = plt.subplots(nrows=rows, ncols=cols)

    for r in range(rows):
        for c in range(cols):
            i = c * rows + r
            ax = axs[r, c]
            ax.imshow(array_to_img(images[i] * 255))
            ax.axis('off')
            ax.set_title(signnames[np.argmax(labels[i])])
            ax.title.set_fontsize(title_font_size)

    plt.show()


def prepare_data_and_model(architecture, method):
    result_folder = os.path.join(RESULT_ROOT, method)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    print("Load and create dataset from file...")
    xtrain, ytrain, xtest, ytest = get_dataset(TRAIN_PATH, TEST_PATH, TEST_LABELS_PATH, (IMG_SIZE, IMG_SIZE))
    num_classes = np.unique(ytrain).size
    # one-hot result vector encoding
    from tensorflow.python.keras.utils import np_utils
    ytrain = np_utils.to_categorical(ytrain, num_classes=num_classes)
    ytest = np_utils.to_categorical(ytest, num_classes=num_classes)

    model_input_shape = xtrain[0].shape
    model = None

    # train model
    if architecture == 'vgg19':
        model = build_vgg19(num_classes, model_input_shape)
    elif architecture == 'lenet-5':
        model = build_lenet5(num_classes, model_input_shape)
    elif architecture == 'alex':
        model = build_alexnet(num_classes, model_input_shape)
    elif architecture == 'resnet50':
        model = build_resnet50(num_classes, model_input_shape)

    train_model(model, xtrain, ytrain, xtest, ytest, modelpath=architecture + "model.h5", result_folder=result_folder)

    return model, xtrain, ytrain, xtest, ytest, result_folder


def build_resnet50(num_classes, img_size):
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Flatten
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=img_size)
    x = Flatten(input_shape=resnet.output.shape)(resnet.output)
    x = Dense(1024, activation='sigmoid')(x)
    predictions = Dense(num_classes, activation='softmax', name='pred')(x)
    model = Model(inputs=[resnet.input], outputs=[predictions])
    return model


def build_vgg19(num_classes, img_size):
    from tensorflow.keras.applications import VGG19
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Flatten
    vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=img_size)
    # customize last layers
    x = Flatten(input_shape=vgg19.output.shape)(vgg19.output)
    x = Dense(1024, activation='sigmoid')(x)
    predictions = Dense(num_classes, activation='softmax', name='pred')(x)
    model = Model(inputs=[vgg19.input], outputs=[predictions])
    # freeze the first 8 layers
    for layer in model.layers[:8]:
        layer.trainable = False
    return model


def build_lenet5(num_classes, img_size):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=img_size))
    model.add(AveragePooling2D())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_alexnet(num_classes, img_size):
    """
    Build an AlexNet
    :param num_classes: number of classes
    :param img_size: image size as tuple (width, height, 3) for rgb and (widht, height) for grayscale
    :return: model
    """
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=img_size,
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def lr_schedule(epoch):
    # decreasing learning rate depending on epoch
    return 0.001 * (0.1 ** int(epoch / 15))#todo: val is 10 not 15


def measure_input_gradient(model, x, y):

    y_placeholder = tf.placeholder(tf.float32, shape=(None, 43))

    epsilon = tf.constant(tf.keras.backend.epsilon(), model.output.dtype.base_dtype)
    output = model(model.input) / tf.reduce_sum(model(model.input), -1, True)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)

    temp = -y_placeholder * tf.log(output)

    categorical_crossentropy = tf.reduce_sum(temp, -1)

    grad = tf.gradients(temp, model.input)[0]

    sum_grad = tf.reduce_sum(grad, [1,2,3])

    sum_grad2 = tf.square(sum_grad)

    sess = tf.keras.backend.get_session()

    batch_size = 512
    ces = []

    for i in range(int(len(x) / batch_size) + 1):
        ces.extend(categorical_crossentropy.eval(session=sess, feed_dict={model.input:   x[i * batch_size:min((i + 1) * batch_size, len(x))],
                                             y_placeholder: y[i * batch_size:min((i + 1) * batch_size, len(x))]}))

    print("avg categorical crossentropy: ", np.average(ces))

    pens = []
    for i in range(int(len(x) / batch_size) + 1):
        pens.extend(sum_grad2.eval(session=sess, feed_dict={model.input:   x[i * batch_size:min((i + 1) * batch_size, len(x))],
                                             y_placeholder: y[i * batch_size:min((i + 1) * batch_size, len(x))]}))
    print("avg input gradient: ", np.average(pens))

    losses = np.multiply(pens, LAMBDA) + ces

    print("avg loss: ", np.average(losses))

    return

# best lambda found: 0.2 (alex)
def get_regularization_loss(model):

    def penalized_loss(target, output):

        # scale preds so that the class probas of each sample sum to 1
        output = output / tf.reduce_sum(output, -1, True)
        epsilon_ = tf.constant(tf.keras.backend.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon_, 1. - epsilon_)

        temp = -target * tf.log(output)

        categorical_crossentropy = tf.reduce_sum(temp, -1) # shape: (?,)

        grad = tf.gradients(categorical_crossentropy, model.input)[0]
        grad2 = tf.square(grad)

        sum_dim = tf.reduce_sum(grad2, [1, 2, 3])

        return categorical_crossentropy + LAMBDA * sum_dim

    return penalized_loss


# def eval_adv_acc(model, x, y):
#     from adversarials import generate_adversarials_fgsm
#
#     SAMPLES = 200
#
#     x_adv = generate_adversarials_fgsm(model, x[:SAMPLES], y[:SAMPLES])
#
#     acc = model.evaluate(x_adv, y[:SAMPLES], batch_size=1024, verbose=0)
#
#     plot_labeled_images(3, 3, x_adv, model.predict(x_adv))
#
#     count = 0
#
#     for i in range(len(x_adv)):
#         count += x_adv[i][0][0][0] != x_adv[i][0][0][0]
#
#     print("not found adv for: ", count)
#
#     return


def train_model(model, xtrain, ytrain, xtest, ytest, modelpath, lr=0.001,
                batch_size=32, epochs=10, result_folder="", adversarial=False):
    """
    Trains a CNN for a given dataset
    :param model: initialized model
    :param xtrain: training images
    :param ytrain: labels for training images numbered from 0 to n
    :param xtest: test images
    :param ytest: labels for test images numbered from 0 to n
    :param lr: initial learning rate for SGD optimizer
    :param batch_size: batch size
    :param epochs: number of epochs to train
    :param result_folder: Save trained model to this directory
    :return: None
    """
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
    from adversarials import get_adversarial_loss

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    loss = get_regularization_loss(model)

    if adversarial:
        # loss = get_adversarial_loss(model, loss)
        loss = get_adversarial_loss(model, loss)

    modelpath = os.path.join(result_folder, modelpath)

    checkpoint = ModelCheckpoint(modelpath, save_best_only=True)
    csv_logger = CSVLogger(os.path.join(result_folder, "training.log"), separator=",", append=True)

    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])

    if os.path.exists(modelpath) and os.path.isfile(modelpath):
        model.load_weights(modelpath)
    else:
        model.fit(xtrain, ytrain,
                  batch_size=batch_size,
                  validation_data=(xtest, ytest),
                  epochs=epochs,
                  callbacks=[LearningRateScheduler(lr_schedule), csv_logger, checkpoint])


# begin experimental code

def create_target_vector(labels):
    first = labels[0]
    labels = labels[1:]
    return np.concatenate((labels, [first]))


def try_attack_params(model, x, y):
    from adversarials import test_FGM_params

    ytarget = np.zeros((1, 43))
    ytarget[0][8] = 1

    eps_iter = 0.02

    while True:
        x_adv = test_FGM_params(model, x[:1], ytarget, eps_iter)

        plt.imshow(x_adv[0])
        plt.show()
        print("label: ", np.argmax(model.predict(x_adv)))

        x[0] = x_adv[0]



# end experimental code

