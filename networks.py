import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img

from preprocessing import get_dataset, signnames

sys.path.append(os.getcwd())

RESULT_ROOT = "results"
TRAIN_PATH = "res\\train\\Final_Training\\Images\\"
TEST_PATH = "res\\test\\Final_Test\\Images\\"
TEST_LABELS_PATH = "res\\test\\Final_Test\\GT-final_test.csv"
IMG_SIZE = 64
EPOCHS_TRAINING_STANDARD = 10
LEARNING_RATE = 0.0005
BATCH_SIZE_TRAINING = 32


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


def prepare_data(experiment):
    result_folder = os.path.join(RESULT_ROOT, experiment)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    print("Load and create dataset from file...")
    xtrain, ytrain, xtest, ytest = get_dataset(TRAIN_PATH, TEST_PATH, TEST_LABELS_PATH, (IMG_SIZE, IMG_SIZE))
    num_classes = np.unique(ytrain).size
    # one-hot result vector encoding
    from tensorflow.python.keras.utils import np_utils
    ytrain = np_utils.to_categorical(ytrain, num_classes=num_classes)
    ytest = np_utils.to_categorical(ytest, num_classes=num_classes)

    return xtrain, ytrain, xtest, ytest, result_folder


def prepare_model_cm(architecture, xtrain, ytrain, xtest, ytest, result_folder, lam, adversarial):
    model_input_shape = xtrain[0].shape
    num_classes = ytrain[0].shape[0]

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

    train_model(model, xtrain, ytrain, xtest, ytest, architecture, lam, adversarial, 0, result_folder)

    return model


def prepare_model_iat(architecture, xtrain, ytrain, xtest, ytest, adversarial):
    model_input_shape = xtrain[0].shape
    num_classes = ytrain[0].shape[0]

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

    train_model_partially(model, xtrain, ytrain, xtest, ytest, adversarial, 0)

    return model


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
    return 0.001 * (0.1 ** int(epoch / EPOCHS_TRAINING_STANDARD))


def get_regularization_loss(model, lam):
    def penalized_loss(target, output):
        categorical_crossentropy = tf.keras.losses.categorical_crossentropy(target, output)

        if lam:
            grad = tf.gradients(categorical_crossentropy, model.input)[0]
            grad2 = tf.square(grad)

            sum_dim = tf.reduce_sum(grad2, [1, 2, 3])

            return categorical_crossentropy + lam * sum_dim

        return categorical_crossentropy

    return penalized_loss


def train_model(model, xtrain, ytrain, xtest, ytest, architecture, lam, adversarial_loss, run, result_folder=""):
    """
    Trains a CNN for a given dataset
    :param model: initialized model
    :param xtrain: training images
    :param ytrain: labels for training images numbered from 0 to n
    :param xtest: test images
    :param ytest: labels for test images numbered from 0 to n
    :param architecture the architecture of the model to train
    :param lam lambda for input gradient regularization
    :param adversarial_loss whether to use FGSM adv loss
    :param run number of this training run
    :param result_folder: Save trained model to this directory
    :return: None
    """
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
    from adversarials import get_adversarial_loss

    adv = "_adv" if adversarial_loss else "_"

    modelpath = architecture + "_lam" + str(lam) + adv + str(run) + ".h5"
    modelpath = os.path.join(result_folder, modelpath)

    checkpoint = ModelCheckpoint(modelpath, save_best_only=True)

    if run == 0:
        loss = get_regularization_loss(model, lam)

        sgd = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)

        if adversarial_loss:
            loss = get_adversarial_loss(model, loss)

        model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])

    if os.path.exists(modelpath) and os.path.isfile(modelpath):
        model.load_weights(modelpath)
    else:
        model.fit(xtrain, ytrain,
                  batch_size=BATCH_SIZE_TRAINING,
                  validation_data=(xtest, ytest),
                  epochs=EPOCHS_TRAINING_STANDARD,
                  callbacks=[LearningRateScheduler(lr_schedule), checkpoint],
                  verbose=0)


def train_model_partially(model, xtrain, ytrain, xtest, ytest, adversarial_loss, run):
    """
    Trains a CNN for a given dataset
    :param model: initialized model
    :param xtrain: training images
    :param ytrain: labels for training images numbered from 0 to n
    :param xtest: test images
    :param ytest: labels for test images numbered from 0 to n
    :param architecture the architecture of the model to train
    :param lam lambda for input gradient regularization
    :param adversarial_loss whether to use FGSM adv loss
    :param run number of this training run
    :param result_folder: Save trained model to this directory
    :return: None
    """
    from tensorflow.keras.optimizers import SGD
    from adversarials import get_adversarial_loss

    epochs_per_run = 2

    if run == 0:
        loss = get_regularization_loss(model, 0)

        if adversarial_loss:
            loss = get_adversarial_loss(model, loss)

        sgd = SGD(lr=LEARNING_RATE, momentum=0.9, nesterov=True)

        model.compile(optimizer=sgd, metrics=['accuracy'], loss=loss)
    else:
        tf.keras.backend.set_value(model.optimizer.lr, lr_schedule(run * epochs_per_run))

    model.fit(xtrain, ytrain,
              batch_size=BATCH_SIZE_TRAINING,
              validation_data=(xtest, ytest),
              epochs=epochs_per_run,
              verbose=1)


def measure_input_gradient(model, x, y):
    y_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, 43))

    epsilon = tf.constant(tf.keras.backend.epsilon(), model.output.dtype.base_dtype)
    output = model(model.input) / tf.reduce_sum(model(model.input), -1, True)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)

    temp = -y_placeholder * tf.math.log(output)

    grad = tf.gradients(temp, model.input)[0]

    sum_grad = tf.reduce_sum(grad, [1, 2, 3])

    sum_grad2 = tf.square(sum_grad)

    sess = tf.compat.v1.keras.backend.get_session()

    batch_size = 128

    input_gradients = []
    for i in range(int(len(x) / batch_size) + 1):
        input_gradients.extend(
            sum_grad2.eval(session=sess, feed_dict={model.input: x[i * batch_size:min((i + 1) * batch_size, len(x))],
                                                    y_placeholder: y[
                                                                   i * batch_size:min((i + 1) * batch_size, len(x))]}))

    print("avg input gradient: ", np.average(input_gradients))
    print("max input gradient: ", np.max(input_gradients))
    return


def evaluate_model(model, x, y):
    from adversarials import get_manipulated_data
    BATCH_SIZE = 128

    _, acc = model.evaluate(x, y, verbose=0, batch_size=BATCH_SIZE)

    print("val acc: ", acc)

    measure_input_gradient(model, x, y)

    adv_fgsm = get_manipulated_data(x, model, 'FGSM')

    _, acc = model.evaluate(adv_fgsm, y, verbose=0, batch_size=BATCH_SIZE)

    print("adv fgsm acc: ", acc)
    # todo: add OPA again
    # OPA_SAMPLES = 128
    #
    # adv_opa, true_labels = get_manipulated_data(x[:OPA_SAMPLES], model, 'OPA', y_original=y[:OPA_SAMPLES])
    #
    # _, acc = model.evaluate(adv_opa, true_labels)
    #
    # print("found one pixel adversarials for ", len(adv_opa), " from ", OPA_SAMPLES, ", rate=",
    #       (OPA_SAMPLES - len(adv_opa)) / OPA_SAMPLES)
    # print("accuracy among OPA - SAMPLES: ", acc)
