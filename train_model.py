import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.evaluation import batch_eval
from tensorflow import keras
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.preprocessing.image import array_to_img

from preprocessing import get_dataset

sys.path.append(os.getcwd())

# MIT-license: https://github.com/MaximilianIdahl/gtsrb-models-keras

model_path = 'results\\model.h5'


def get_adversarial_acc_metric(model, fgsm, fgsm_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_acc


def get_adversarial_loss(model, fgsm, fgsm_params):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        return 0.5 * cross_ent + 0.5 * cross_ent_adv

    return adv_loss

def get_signname_dict():
    import csv
    with open("res\\signnames.csv") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        i = 0

        result = {}

        for line in reader:
            if i > 0:
                result[i - 1] = line[1]
            i += 1

    return result


def plot_labeled_images(rows, cols, images, labels, gray):
    title_font_size = 8

    signnames = get_signname_dict()

    fig, axs = plt.subplots(nrows=rows, ncols=cols)

    for r in range(rows):
        for c in range(cols):
            i = c * rows + r
            ax = axs[r, c]
            if gray:
                ax.imshow(array_to_img(images[i] * 255), cmap='gray')
            else:
                ax.imshow(array_to_img(images[i] * 255))
            ax.axis('off')
            ax.set_title(signnames[np.argmax(labels[i])])
            ax.title.set_fontsize(title_font_size)

    plt.show()


def plot(history, result_folder):
    # Plot training & validation accuracy values
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(["Train (max = {0:.2f})".format(100 * max(history.history["acc"])),
                "Test (max = {0:.2f})".format(100 * max(history.history["val_acc"]))], loc='upper left')
    plt.grid(True)
    plt.xticks(range(0, len(history.history['acc'])))
    plt.savefig(os.path.join(result_folder, "model_acc.svg"), format="svg", dpi=1000, bbox_inches='tight')

    # Plot training & validation loss values
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.xticks(range(0, len(history.history['acc'])))
    plt.savefig(os.path.join(result_folder, "model_loss.svg"), format="svg", dpi=1000, bbox_inches='tight')


def build_resnet50(num_classes, img_size):
    from tensorflow.python.keras.applications import ResNet50
    from tensorflow.python.keras import Model
    from tensorflow.python.keras.layers import Dense, Flatten
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=img_size)
    x = Flatten(input_shape=resnet.output.shape)(resnet.output)
    x = Dense(1024, activation='sigmoid')(x)
    predictions = Dense(num_classes, activation='softmax', name='pred')(x)
    model = Model(inputs=[resnet.input], outputs=[predictions])
    return model


def build_vgg19(num_classes, img_size):
    from tensorflow.python.keras.applications import VGG19
    from tensorflow.python.keras import Model
    from tensorflow.python.keras.layers import Dense, Flatten
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
    from tensorflow.python.keras import Sequential
    from tensorflow.python.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
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
    from tensorflow.python.keras import Sequential
    from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
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
    return 0.001 * (0.1 ** int(epoch / 10))


def train_model(model, xtrain, ytrain, xtest, ytest, load = False, lr=0.001,
                batch_size=32, epochs=10, result_folder=""):
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
    from tensorflow.python.keras.optimizers import SGD
    from tensorflow.python.keras.callbacks import LearningRateScheduler, ModelCheckpoint
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    loss = 'categorical_crossentropy'

    checkpoint = ModelCheckpoint(os.path.join(result_folder, "model.h5"), save_best_only=True)
    csv_logger = CSVLogger(os.path.join(result_folder, "training.log"), separator=",", append=True)

    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])

    if load:
        model.load_weights(model_path)
    else:
        history = model.fit(xtrain, ytrain,
            batch_size=batch_size,
            validation_data=(xtest, ytest),
            epochs=epochs,
            callbacks=[LearningRateScheduler(lr_schedule), csv_logger, checkpoint])

    # plot(history, result_folder)


def generate_adversarial_inputs(x_in, fgsm, fgsm_params):
    x_in = x_in.astype('float32') # todo: exchange float64

    x = tf.placeholder(tf.float32, shape=(None, 64, 64, 3))

    x_adv = fgsm.generate(x, **fgsm_params)
    x_adv = tf.stop_gradient(x_adv)
    eval_params = {'batch_size': 128}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        adv_images = batch_eval(sess, [x], [x_adv], [x_in], args=eval_params)[0]

    return adv_images


def main(architecture, train_path, test_path, test_labels_path, img_size, result_folder, grayscale):

    print("Load and create dataset from file...")
    xtrain, ytrain, xtest, ytest = get_dataset(train_path, test_path, test_labels_path, (img_size, img_size), grayscale)
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

    train_model(model, xtrain, ytrain, xtest, ytest)

    model(model.input)

    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap)
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}

    x_adv_series = []

    for i in range(100):

        x_adv_train = generate_adversarial_inputs(xtrain, fgsm, fgsm_params).astype('float64')
        x_adv_test = generate_adversarial_inputs(xtest, fgsm, fgsm_params).astype('float64')

        x_adv_series.extend([x_adv_test[0]])

        xtrain = np.concatenate([xtrain, x_adv_train])
        ytrain = np.concatenate([ytrain, ytrain])

        train_model(model, xtrain, ytrain, xtest, ytest, lr=0.001, batch_size=32, epochs=10,
                result_folder=result_folder)

        print("accuracy on test examples: ", model.evaluate(xtest, ytest, 1024, 0))
        print("accuracy on adversary test examples: ", model.evaluate(x_adv_test, ytest, 1024, 0))

        y_adv = model.predict(x_adv_test, batch_size=1024)

        plot_labeled_images(3, 3, xtest, ytest, grayscale)
        plot_labeled_images(3, 3, x_adv_test, y_adv, grayscale)

        xtrain = xtrain[0:int(len(xtrain) / 2)]
        ytrain = ytrain[0:int(len(ytrain) / 2)]

        wrap = KerasModelWrapper(model)
        fgsm = FastGradientMethod(wrap)


if __name__ == "__main__":
    import sys, argparse

    argv = sys.argv[1:]
    usage_text = "Run as " + __file__ + " [options]"
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument(
        "-a", "--architecture", dest="architecture", required=True,
        choices=['alex', 'vgg19', 'resnet50', 'lenet-5'],
        help="model architecture for training. \nOptions: ['alex', 'vgg19', 'resnet50', 'lenet-5']"
    )
    parser.add_argument(
        "-t", "--train_path", dest="train_path", type=str, required=False,
        default="res\\train\\Final_Training\\Images\\",
        help="Input directory for train set"
    )
    parser.add_argument(
        "-v", "--validation_path", dest="validation_path", type=str, required=False,
        help="Input directory for test/validation set",
        default="res\\test\\Final_Test\\Images\\"
    )
    parser.add_argument(
        "-l", "--validation_labels", dest="validation_labels", type=str, required=False,
        help="Path to GT-final_test.csv file",
        default="res\\test\\Final_Test\\GT-final_test.csv"
    )
    parser.add_argument(
        "-i", "--img_dim", dest="img_dim", type=int, required=False,
        default=64,
        help="Image width and height in pixel"
    )
    parser.add_argument(
        "-r", "--result_folder", dest="result_folder", type=str, required=False,
        default="results/",
        help="Output directory"
    )


    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser.add_argument(
        "-g", "--grayscale", dest="gs", type=str2bool, default="false", help="Train only on grayscale images"
    )



    if not argv:
        print("Some required argument is missing")
    args = parser.parse_args(argv)
    # result directory
    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)
    if not os.path.exists(args.train_path):
        raise IOError("Cannot read directory {}".format(args.train_path))
    if not os.path.exists(args.train_path):
        raise IOError("Cannot read directory {}".format(args.validation_path))
    if not os.path.exists(args.train_path):
        raise IOError("Cannot read file {}".format(args.validation_labels))

    main(args.architecture,
         args.train_path,
         args.validation_path,
         args.validation_labels,
         args.img_dim,
         args.result_folder,
         args.gs)