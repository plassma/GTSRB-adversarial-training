from networks import *
from report import Report
from adversarials import get_manipulated_data


def confusion_matrix(labels, architecture, method, epsilon, iterations, adversarial):
    # example for labels: "0,1,2,3" or "12, 13, 14, 15"
    labels = [int(s) for s in labels.replace(" ", "").split(",")]

    model, xtrain, ytrain, xtest, ytest, result_folder = prepare_data_and_model(architecture, method, adversarial)

    evaluate_model(model, xtest, ytest)

    plot_confusion_matrix(labels, model, xtest, ytest, epsilon, iterations)


def create_target_vector(labels):
    first = labels[0]
    labels = labels[1:]
    return np.concatenate((labels, [first]))


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

        train_model(model, xtrain, ytrain, xtest, ytest, architecture, run + 1, True,
                    result_folder=result_folder)

        report.add_data_after(model, xtest, x_adv_test)

        report.report()


def plot_confusion_matrix(labels, model, x, y, eps, iterations):
    from adversarials import transform_to_target_BIM
    from preprocessing import signnames

    n = len(labels)

    preds = model.predict(x, batch_size=1024)

    input_images = []
    targets = []
    for label in labels:
        for i in range(len(x)):
            if label == np.argmax(y[i]) and label == np.argmax(preds[i]):
                input_images.append(np.repeat([x[i]], n, 0))
                targets.append(y[i])
                break

    input_images = np.array(input_images)
    targets = np.array(targets)

    fig, axs = plt.subplots(nrows=n, ncols=n, figsize=(10, 10))

    plt.text(1, 1.3, "Target Label", horizontalalignment='center',
             fontsize=20,
             transform=axs[0, 1].transAxes)

    plt.text(-0.4, 0, "Original Label", verticalalignment='center',
             fontsize=20,
             rotation='vertical',
             transform=axs[1, 0].transAxes)

    for r in range(n):
        adv_x = transform_to_target_BIM(model, input_images[r], targets, eps, iterations)
        for c in range(n):
            ax = axs[r, c]

            if r == 0:
                ax.set_title(signnames[labels[c]])
            if c == 0:
                ax.set_ylabel(signnames[labels[r]])

            ax.set_xticks([])
            ax.set_yticks([])

            if c != r:
                ax.imshow(adv_x[c])
            else:
                ax.imshow(input_images[r, c])
    plt.show()
