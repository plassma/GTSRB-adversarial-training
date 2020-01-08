import warnings
from experiments import perform_iterative_adversarial_training
from networks import prepare_data

warnings.filterwarnings("ignore")


data_tuple = prepare_data("iterative_adversarial_training")

# perform_iterative_adversarial_training("lenet-5", data_tuple)
# perform_iterative_adversarial_training("alex", data_tuple)
# perform_iterative_adversarial_training("vgg19", data_tuple)
perform_iterative_adversarial_training("resnet50", data_tuple)
