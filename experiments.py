from networks import *
from report import Report
from adversarials import get_manipulated_data


def confusion_matrix(labels, architecture, method):
    # example for labels: [0,1,2,3]
    model, xtrain, ytrain, xtest, ytest, result_folder = prepare_data_and_model(architecture, method)

    plot_confusion_matrix(labels, model, xtest, ytest)


def iterative_adversarial_training(architecture, method, iterations=10, targeted=False):

    model, xtrain, ytrain, xtest, ytest, result_folder = prepare_data_and_model(architecture, method)

    # To be able to call the model in the custom loss, we need to call it once
    # before, see https://github.com/tensorflow/tensorflow/issues/23769
    model(model.input)

    report = Report(result_folder, architecture)

    for run in range(iterations):

        y_adv_test_target = create_target_vector(ytest) if targeted else None

        x_adv_test = get_manipulated_data(xtest, model, method, y_adv_test_target,
                                          result_folder, "advtest", architecture, run)

        report.evaluate_accuracies(model, xtest, ytest, architecture, method, run)

        # y_adv_train_target = create_target_vector(ytrain) if targeted else None
        # x_adv_train = get_manipulated_data(xtrain, model, method, y_adv_train_target,
        #                                   result_folder, "advtrain", architecture, run)

        y_adv_test = model.predict(x_adv_test)

        report.add_data_before(model, xtest, ytest, x_adv_test, y_adv_test)

        train_model(model, xtrain, ytrain, xtest, ytest, modelpath=architecture + str(run) + "_adv.h5",
                    result_folder=result_folder, adversarial=True)

        report.add_data_after(model, xtest, x_adv_test)

        report.report()


def plot_confusion_matrix(labels, model, x, y, reps=10, eps=0.03):
    from adversarials import test_FGM_params

    n = len(labels)

    preds = model.predict(x, batch_size=1024)

    inputs = []
    targets = []
    for label in labels:
        for i in range(len(x)):
            if label == np.argmax(y[i]) and label == np.argmax(preds[i]):
                inputs.append(np.repeat([x[i]], n, 0))
                targets.append(y[i])
                break

    inputs = np.array(inputs)
    targets = np.array(targets)

    fig, axs = plt.subplots(nrows=n, ncols=n)

    for r in range(n):
        adv_x = inputs[r]
        for i in range(reps):
            adv_x = test_FGM_params(model,  adv_x, targets, eps)
        for c in range(n):
            ax = axs[r, c]
            ax.axis('off')

            if c != r:
                ax.imshow(adv_x[c])
            else:
                ax.imshow(inputs[r, c])
    plt.show()
