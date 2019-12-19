import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img

from adversarials import get_manipulated_data
from heatmap import grad_CAM_plus
from preprocessing import signnames

SAMPLES = 100


class Report:
    def __init__(self, result_folder, architecture):

        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        self.result_folder = result_folder
        self.run = 0
        self.architecture = architecture
        self.csv_file = open(os.path.join(result_folder, architecture, "accuracies.csv"), "w")
        self.accuracy_csv_writer = csv.writer(self.csv_file)

        self.data = {}

    @staticmethod
    def to_index(one_hot_arr):
        result = np.zeros(one_hot_arr.shape[0])

        for i in range(one_hot_arr.shape[0]):
            result[i] = np.argmax(one_hot_arr[i])
        return result

    def add_data_before(self, model, x, y, x_adv, y_adv):
        self.data['x_orig'] = [(x, np.argmax(y)) for x, y in zip(x[0:SAMPLES], y[0:SAMPLES])]
        self.data['x_adv'] = [(x, np.argmax(y)) for x, y in zip(x_adv[0:SAMPLES], y_adv[0:SAMPLES])]

        self.data['x_orig_heat_before'] = grad_CAM_plus(model, x[:SAMPLES])
        self.data['x_adv_heat'] = grad_CAM_plus(model, x_adv[:SAMPLES])

    def add_data_after(self, model, x, x_adv):
        self.data['x_orig_heat_after'] = grad_CAM_plus(model, x[:SAMPLES])
        self.data['x_adv_heat_after'] = grad_CAM_plus(model, x_adv[:SAMPLES])

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

    def report(self):

        result_folder = os.path.join(self.result_folder, self.architecture, str(self.run))

        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        for i in range(len(self.data['x_orig'])):
            fig, axs = plt.subplots(nrows=2, ncols=3)
            self.show_tuple(self.data['x_orig'][i], 'original', '', axs[0, 0])
            self.show_tuple(self.data['x_adv'][i], 'adversarial', '', axs[1, 0])

            self.show_tuple(self.data['x_orig_heat_before'][i], 'original', 'before adv-training', axs[0, 1])
            self.show_tuple(self.data['x_adv_heat'][i], 'adversarial', 'before adv-training', axs[1, 1])

            self.show_tuple(self.data['x_orig_heat_after'][i], 'original', 'after adv-training', axs[0, 2])
            self.show_tuple(self.data['x_adv_heat_after'][i], 'adversarial', 'after adv-training', axs[1, 2])

            plt.savefig(os.path.join(result_folder, str(i) + ".png"))
            plt.close()

        self.run += 1

    def evaluate_accuracies(self, model, xtest, ytest, architecture, method, run):

        accuracies = [run, model.evaluate(xtest, ytest, batch_size=1024, verbose=0)[1]]

        for i in range(run + 1):
            adv = get_manipulated_data(None, model, method, None, self.result_folder, "advtest", architecture, i)
            _, acc = model.evaluate(adv, ytest, batch_size=1024, verbose=0)
            accuracies.append(acc)

        self.accuracy_csv_writer.writerow(accuracies)
        self.csv_file.flush()