from adversarial_timelapse import AdversarialTimelapse
import numpy as np
from networks import prepare_data


def plot_adversarial_timelapse(architecture, adv):
    base_folder = "results/iterative_adversarial_training" + ("_adv/" if adv else "/") + architecture + "/advtest/"
    adversarial_timelase = AdversarialTimelapse(xtest)

    for i in range(0, 10):
        file = base_folder + str(i) + "/data.npy"

        adversarials = np.load(file)
        adversarial_timelase.add_adversarials(adversarials)

    adversarial_timelase.plot_timelapse(architecture, adv)


xtrain, ytrain, xtest, ytest, result_folder = prepare_data('None')

plot_adversarial_timelapse('lenet-5', False)
plot_adversarial_timelapse('lenet-5', True)

plot_adversarial_timelapse('alex', False)
plot_adversarial_timelapse('alex', True)

plot_adversarial_timelapse('vgg19', False)
plot_adversarial_timelapse('vgg19', True)

plot_adversarial_timelapse('resnet50', False)
plot_adversarial_timelapse('resnet50', True)