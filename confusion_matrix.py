import warnings

from experiments import perform_confusion_matrix
from networks import prepare_data

warnings.filterwarnings("ignore")

data_tuple = prepare_data("confusion_matrix")

perform_confusion_matrix('lenet-5', data_tuple, 0.4)
perform_confusion_matrix('alex', data_tuple, 0.15)
perform_confusion_matrix('vgg19', data_tuple, 0.002) #todo: try 0.003
perform_confusion_matrix('resnet50', data_tuple, 0.002) #todo: find good value
