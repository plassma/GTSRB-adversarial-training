from experiments import prepare_data, prepare_model_cm
from adversarials import generate_adversarials_OPA
import matplotlib.pyplot as plt
import numpy as np


def eval_improvement(architecture):
    model = prepare_model_cm(architecture, xtrain, ytrain, xtest, ytest, result_folder, 0, False)
    adversarials_before = generate_adversarials_OPA(model, xtest, ytest)
    model = prepare_model_cm(architecture, xtrain, ytrain, xtest, ytest, result_folder, 0, True)
    adversarials_after = generate_adversarials_OPA(model, xtest, ytest)
    print(architecture)
    print(f"Adversarials found before: {len(adversarials_before[0])}, Rate: {len(adversarials_before[0]) / len(xtest)}")
    print(f"Adversarials found after: {len(adversarials_after[0])}, Rate: {len(adversarials_after[0]) / len(xtest)}")
    print("\n\n")

xtrain, ytrain, xtest, ytest, result_folder = prepare_data('confusion_matrix')

#eval_improvement('lenet-5')
#eval_improvement('alex')
eval_improvement('vgg19')
