from networks import *
from report import Report
from adversarials import get_manipulated_data

THESIS_PATH = "Thesis\\graphics\\Results\\confusion_matrix\\"


def perform_confusion_matrix(architecture, data_tuple, lam):
    EPSILON_BIM = 0.03
    ITERATIONS_BIM = 15

    confusion_matrix(architecture, data_tuple, EPSILON_BIM, ITERATIONS_BIM, 0, False)
    confusion_matrix(architecture, data_tuple, EPSILON_BIM, ITERATIONS_BIM, lam, False)
    confusion_matrix(architecture, data_tuple, EPSILON_BIM, ITERATIONS_BIM, 0, True)
    confusion_matrix(architecture, data_tuple, EPSILON_BIM, ITERATIONS_BIM, lam, True)


def confusion_matrix(architecture, data_tuple, epsilon, iterations, lam, adversarial):
    LABELS_SIMILAR = [0, 1, 2, 3]
    LABELS_DIFFERENT = [12, 13, 14, 15]

    xtrain, ytrain, xtest, ytest, result_folder = data_tuple

    print("Performing for ", architecture, " lambda=", lam, ", adversarial=", str(adversarial))
    print("Attack-params: eps: ", epsilon, ", iterations: ", iterations)

    model = prepare_model(architecture, xtrain, ytrain, xtest, ytest, result_folder, lam, adversarial)

    evaluate_model(model, xtest, ytest)

    text = architecture + r", $\lambda$: " + str(lam) + r", $\alpha$: " + str(0.5 if adversarial else 0) + \
        r", $\epsilon_{BIM}: $" + str(epsilon) + r", $iter_{BIM}$:" + str(iterations)

    export_to_thesis = False
    plot_path = os.path.join(THESIS_PATH,
                             architecture + ("_gradreg" if lam > 0 else "") + ("_adv" if adversarial else ""))

    os.makedirs(plot_path, exist_ok=True)

    plot_confusion_matrix(LABELS_SIMILAR, model, xtest, ytest, epsilon, iterations, text)

    if export_to_thesis:
        plt.savefig(os.path.join(plot_path, "similar.png"))
    else:
        plt.show()

    plot_confusion_matrix(LABELS_DIFFERENT, model, xtest, ytest, epsilon, iterations, text)

    if export_to_thesis:
        plt.savefig(os.path.join(plot_path,
                                "different.png"))
    else:
        plt.show()


def create_target_vector(labels):
    first = labels[0]
    labels = labels[1:]
    return np.concatenate((labels, [first]))


def transform_to_batch_acc(accuracies):
    batches = []

    l = len(accuracies)

    for i in range(l):
        batch = []
        for j in range(l):
            if j >= len(accuracies[i]):
                batch.append(float("nan"))
            else:
                batch.append(accuracies[i][j])
        batches.append(batch)

    return np.array(batches)


def perform_iterative_adversarial_training(architecture, data_tuple):
    accuracies_adv_data = iterative_adversarial_training(architecture, data_tuple, False, 3)
    accuracies_adv_loss = iterative_adversarial_training(architecture, data_tuple, True, 3)

    accuracies_adv_data_batches = transform_to_batch_acc(accuracies_adv_data)
    accuracies_adv_loss_batches = transform_to_batch_acc(accuracies_adv_loss)

    for i in range(len(accuracies_adv_loss)):
        plt.plot(accuracies_adv_data_batches[i])
    plt.plot()

    for i in range(len(accuracies_adv_loss)):
        plt.plot(accuracies_adv_loss_batches[i])
    plt.plot()


def iterative_adversarial_training(architecture, data_tuple, use_adv_loss=True, iterations=10, targeted=False):
    xtrain, ytrain, xtest, ytest, result_folder = data_tuple

    model = prepare_model(architecture, xtrain, ytrain, xtest, ytest, result_folder, 0, False)

    # To be able to call the model in the custom loss, we need to call it once
    # before, see https://github.com/tensorflow/tensorflow/issues/23769
    model(model.input)

    report = Report(result_folder, architecture)

    for run in range(iterations):
        y_adv_test_target = create_target_vector(ytest) if targeted else None

        x_adv_test = get_manipulated_data(xtest, model, "FGSM", None, y_adv_test_target,
                                          result_folder, "advtest", architecture, run)

        report.evaluate_accuracies(model, xtest, ytest, architecture, "FGSM", run)

        report.add_data_before(model, xtest, ytest, x_adv_test)

        if not use_adv_loss:
            y_adv_train_target = create_target_vector(ytrain) if targeted else None
            x_adv_train = get_manipulated_data(xtrain, model, "FGSM", None, y_adv_train_target,
                                               result_folder, "advtrain", architecture, run)

            xtrain = np.concatenate((xtrain, x_adv_train))
            ytrain = np.concatenate((ytrain, ytrain))

        train_model(model, xtrain, ytrain, xtest, ytest, architecture, 0, use_adv_loss, run + 1,
                    result_folder=result_folder)

        report.add_data_after(model, xtest, x_adv_test)

        report.report()

        if not use_adv_loss:
            xtrain = xtrain[:int(len(xtrain) / 2)]
            ytrain = ytrain[:int(len(ytrain) / 2)]

    return report.accuracies


def plot_confusion_matrix(labels, model, x, y, eps, iterations, text):
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

    plt.text(0, -0.3, text, fontsize=20, transform=axs[3, 0].transAxes)

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
