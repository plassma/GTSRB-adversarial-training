import os

import numpy as np
import tensorflow as tf


# params found for FastGradientMethod: eps=0.03

def get_manipulated_data(x, model, method, y_original=None, y_target=None, cache_path=None, dataset=None, architecture=None, run=None):
    from pathlib import Path
    fileame = "data.npy"

    if cache_path:
        cache_path = Path(cache_path, architecture, dataset, str(run), fileame)
        if cache_path.exists():
            return np.load(str(cache_path))

    if method == 'FGSM':
        result = generate_adversarials_fgsm_cleverhans(model, x, y_target)
    elif method == 'CWL2':
        result = generate_adversarials_cwl2(model, x, y_target)
    elif method == 'GAUSS':
        result = generate_gaussian_noise(x, 0.03)
    elif method == 'OPA':
        result = generate_adversarials_OPA(model, x, y_original)
    else:
        raise Exception("Method " + method + " not implemented")

    if cache_path:
        os.makedirs(str(cache_path.parent), exist_ok=True)
        np.save(str(cache_path), result)

    return result


def transform_to_target_BIM(model, x, y_target, eps, iterations):
    from cleverhans.attacks import BasicIterativeMethod
    from cleverhans.utils_keras import KerasModelWrapper

    x_placeholder = tf.placeholder(tf.float32, shape=(None, 64, 64, 3))

    wrap = KerasModelWrapper(model)
    attack = BasicIterativeMethod(wrap)
    attack_params = {'eps_iter': eps,
                     'nb_iter': iterations,
                     'clip_min': 0.,
                     'clip_max': 1.,
                     'y_target': y_target}

    x_adv = attack.generate(x_placeholder, **attack_params)

    sess = tf.keras.backend.get_session()

    adv_images = x_adv.eval(session=sess, feed_dict={x_placeholder: x})

    return np.array(adv_images)


def get_adversarial_loss(model, loss_function):
    from cleverhans.attacks import FastGradientMethod
    from cleverhans.utils_keras import KerasModelWrapper

    # turn off learning phase to prevent dropout from destroying gradients
    tf.keras.backend.set_learning_phase(False)

    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap)

    fgsm_params = {'eps': 0.05,
                   'clip_min': 0.,
                   'clip_max': 1.}

    def regularized_adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = loss_function(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = loss_function(y, preds_adv)

        return 0.5 * cross_ent + 0.5 * cross_ent_adv

    # turn learning phase back on
    tf.keras.backend.set_learning_phase(True)

    return regularized_adv_loss


def generate_adversarials_OPA(model, x, y):
    from foolbox.v1.attacks import SinglePixelAttack
    from foolbox.models import KerasModel

    advs = []
    true_labels = []

    keras_model = KerasModel(model, bounds=(0, 1))

    attack = SinglePixelAttack(keras_model)

    for i in range(len(x)):
        img = attack(x[i], np.argmax(y[i]))
        if np.any(img):
            advs.append(img)
            true_labels.append(y[i])

    return np.array(advs), np.array(true_labels)


def generate_adversarials_fgsm_cleverhans(model, x, y_target=None):
    from cleverhans.attacks import FastGradientMethod
    from cleverhans.utils_keras import KerasModelWrapper

    x_placeholder = tf.placeholder(tf.float32, shape=(None, 64, 64, 3))
    y_target_placeholder = tf.placeholder(tf.float32, shape=(None, 43))

    wrap = KerasModelWrapper(model)
    attack = FastGradientMethod(wrap)
    attack_params = {'eps': 0.05,
                     'clip_min': 0.,
                     'clip_max': 1.,
                     'y_target': y_target_placeholder if y_target else None}

    x_adv = attack.generate(x_placeholder, **attack_params)
    x_adv = tf.stop_gradient(x_adv)

    batch_size = 1024

    adv_images = []

    sess = tf.keras.backend.get_session()

    for i in range(int(len(x) / batch_size) + 1):
        print("generating adversarials batch", i, "/", int(len(x) / batch_size))
        start = i * batch_size
        end = min((i + 1) * batch_size, len(x))

        feed_dict = {x_placeholder: x[start:end]}
        if y_target:
            feed_dict[y_target_placeholder] = y_target[start:end]

        adv_images.extend(x_adv.eval(session=sess,
                                     feed_dict=feed_dict))

    return np.array(adv_images)


def generate_adversarials_cwl2(model, x, y_target=None):
    from cleverhans.attacks import CarliniWagnerL2
    from cleverhans.utils_keras import KerasModelWrapper

    wrap = KerasModelWrapper(model)

    cw_params = {'binary_search_steps': 5,
                 'max_iterations': 1000,
                 'learning_rate': .9,
                 'batch_size': len(x),
                 'initial_const': 10}

    with tf.Session() as sess:
        attack = CarliniWagnerL2(wrap, sess)

        result = attack.generate(tf.convert_to_tensor(x.astype('float32')), **cw_params)
        sess.run(tf.global_variables_initializer())
        result = result.eval()

    return result


def generate_gaussian_noise(x, eps):
    result = np.copy(x)

    for i in range(len(result)):
        result[i] += np.random.normal(loc=0, scale=eps, size=result[i].shape)

    return np.clip(result, 0, 1.0)
