import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img

from adversarials import get_manipulated_data
from heatmap import grad_CAM_plus
from preprocessing import signnames

SAMPLES = 25
BATCH_SIZE = 512


class Report:
    def __init__(self, result_folder, architecture):

        if not os.path.exists(os.path.join(result_folder, architecture)):
            os.makedirs(os.path.join(result_folder, architecture), exist_ok=True)

        self.result_folder = result_folder
        self.run = 0
        self.architecture = architecture
        self.csv_file = open(os.path.join(result_folder, architecture, "accuracies.csv"), "w")
        self.accuracy_csv_writer = csv.writer(self.csv_file)
        self.accuracies = []

        self.data = {}

    @staticmethod
    def to_index(one_hot_arr):
        result = np.zeros(one_hot_arr.shape[0])

        for i in range(one_hot_arr.shape[0]):
            result[i] = np.argmax(one_hot_arr[i])
        return result

    @staticmethod
    def show_tuple(im_pred, xlabel, ylabel, ax):
        im, pred = im_pred
        ax.imshow(array_to_img(im))
        title = signnames[pred]

        if len(title) > 25:
            title = title[:25] + '\n' + title[25:]

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_yticks([])

    def report(self, model, x):

        result_folder = os.path.join(self.result_folder, self.architecture, str(self.run))

        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        heatmaps = grad_CAM_plus(model, x[:SAMPLES])
        plt.axis('off')
        for i in range(len(heatmaps)):
            plt.imsave(os.path.join(result_folder, str(i)), heatmaps[i][0])
        plt.axis('on')

        self.run += 1

    def evaluate_accuracies(self, model, xtest, ytest, architecture, method, run):

        accuracies = [run, model.evaluate(xtest, ytest, batch_size=BATCH_SIZE, verbose=0)[1]]

        for i in range(run + 1):
            adv = get_manipulated_data(None, model, method, None, None, self.result_folder, "advtest", architecture, i)
            _, acc = model.evaluate(adv, ytest, batch_size=BATCH_SIZE, verbose=0)
            accuracies.append(acc)
        self.accuracies.append(accuracies[1:])
        self.accuracy_csv_writer.writerow(accuracies)
        self.csv_file.flush()
