import warnings

from experiments import prepare_model_cm, prepare_data
import numpy as np

warnings.filterwarnings("ignore")


def accuracy_against(architecture, adv, ytest, model):
    adversarials_path = "results/iterative_adversarial_training" + \
                                  ("_adv/" if adv else "/") + architecture + "/advtest/9/data.npy"
    adversarials = np.load(adversarials_path)

    return model.evaluate(adversarials, ytest, verbose=0)[1]


def cross_adv_robustness(model, architecture, adv, other_architecture, other_adv, ytest):

    acc = accuracy_against(other_architecture, other_adv, ytest, model)

    print("Accuracy of " + architecture + (" with" if adv else " without") + " adversarial training against adversarials of " +
          other_architecture + (" with" if other_adv else " without") + " adversarial training: " + str(acc))
    return acc


def cross_adv_robustness_csv(architectures, data_tuple):

    xtrain, ytrain, xtest, ytest, result_folder = data_tuple
    csv_file = open("results/cross_adv_robustness/cross_robustness.csv", "w")

    csv_file.write("architecture/adv;")
    for col_architecture in architectures:
        for col_adv in [False, True]:
            csv_file.write(col_architecture + "/" + str(col_adv) + ";")
    csv_file.write("\n")

    for row_architecture in architectures:
        for row_adv in [False, True]:
            model = prepare_model_cm(row_architecture, xtrain, ytrain, xtest, ytest, result_folder, 0, row_adv)
            csv_file.write(row_architecture + "/" + str(row_adv) + ";")
            for col_architecture in architectures:
                for col_adv in [False, True]:
                    acc = cross_adv_robustness(model, row_architecture, row_adv, col_architecture, col_adv, ytest)
                    csv_file.write(str(acc) + ";")
            csv_file.write("\n")

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

#data_tuple = prepare_data("confusion_matrix")

#cross_adv_robustness_csv(["lenet-5", "alex", "vgg19"], data_tuple)
import csv

csv_reader = csv.reader(open("results/cross_adv_robustness/cross_robustness.csv", "r"), delimiter=';')
outfile = open("results/cross_adv_robustness/cross_robustness_rounded.csv", "w")
for row in csv_reader:
    for val in row:
        val = val.strip()
        if isfloat(val):
            outfile.write(str(round(float(val), 2)) + ";")
        else:
            outfile.write(val + ";")
    outfile.write("\n")