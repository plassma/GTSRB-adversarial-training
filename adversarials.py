import os

import numpy as np
import tensorflow as tf


# params found for FastGradientMethod: eps=0.03

def manipulate_data(x, model, method, cache_path, y_target=None):
    if method == 'FGSM':
        return generate_adversarials_cleverhans(model, x, y_target)
    elif method == 'CWL2':
        return generate_adversarials_CLW2_art(model, x, y_target)
    else:
        return add_gaussian_noise(x, 0.03)


def test_FGM_params(model, x, y_target, eps=0.1):
    from cleverhans.attacks import FastGradientMethod
    from cleverhans.utils_keras import KerasModelWrapper

    x_placeholder = tf.placeholder(tf.float32, shape=(None, 64, 64, 3))

    wrap = KerasModelWrapper(model)
    attack = FastGradientMethod(wrap)
    attack_params = {'eps': eps,
                     'clip_min': 0.,
                     'clip_max': 1.,
                     'y_target': y_target}

    x_adv = attack.generate(x_placeholder, **attack_params)

    sess = tf.keras.backend.get_session()

    adv_images = x_adv.eval(session=sess, feed_dict={x_placeholder: x})

    return np.array(adv_images)


def generate_adversarials_cleverhans(model, x, y_target=None):
    from cleverhans.attacks import BasicIterativeMethod
    from cleverhans.utils_keras import KerasModelWrapper

    x_placeholder = tf.placeholder(tf.float32, shape=(None, 64, 64, 3))

    wrap = KerasModelWrapper(model)
    attack = BasicIterativeMethod(wrap)
    attack_params = {'eps_iter': 0.001,
                     'nb_iter': 1,
                     'clip_min': 0.,
                     'clip_max': 1.,
                     'y_target': y_target}

    x_adv = attack.generate(x_placeholder, **attack_params)
    batch_size = 256

    adv_images = []

    sess = tf.keras.backend.get_session()

    for i in range(int(len(x) / batch_size) + 1):
        adv_images.extend(x_adv.eval(session=sess,
                                     feed_dict={x_placeholder: x[i * batch_size:min((i + 1) * batch_size, len(x))]}))

    return np.array(adv_images)


def generate_adversarials_foolbox(model, x, y):
    import foolbox

    result = np.empty(x.shape)

    with tf.compat.v1.keras.backend.get_session().as_default():
        fmodel = foolbox.models.KerasModel(model, bounds=(0,255))
        attack = foolbox.v1.attacks.FGSM(fmodel)

        for i in range(0, len(x)):
            result[i] = attack(x[i], np.argmax(y[i]), epsilons=[0.0002])
    return result


## todo: repair caching
def generate_adversarials_FGSM(model, x, y_target=None, cache_path=None):
    from art.attacks import FastGradientMethod
    from art.classifiers import KerasClassifier

    if cache_path is not None and os.path.exists(os.path.join(cache_path, "data.npy")):
        return load_adversarials(cache_path)

    if cache_path is not None:
        os.makedirs(cache_path)

    classifier = KerasClassifier(model=model)

    attack = FastGradientMethod(classifier=classifier, eps=0.03, batch_size=1024,
                                targeted=y_target is not None)

    result = attack.generate(x, y_target)

    if cache_path is not None:
        save_adversarials(cache_path, result)

    return result


def generate_adversarials_CWL2(model, x, y_target=None):
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


def generate_adversarials_CLW2_art(model, x, y_target=None):
    # params found for alex: conf=0.9, lr=0.01, initial_const=0.03, bin_search_steps=20, max_iter=2
    from art.attacks import CarliniL2Method
    from art.classifiers import KerasClassifier

    classifier = KerasClassifier(model)

    attack = CarliniL2Method(classifier, 0.9, learning_rate=0.02, initial_const=0.03, binary_search_steps=20,
                             max_iter=2, batch_size=1024)

    result = attack.generate(x)

    abs_diff = np.abs(result - x)
    print(abs_diff.min(), abs_diff.mean(), abs_diff.max())

    return result

def generate_adversarials_CLW2_foolbox(model, x, y):
    from foolbox.attacks import CarliniWagnerL2Attack
    from foolbox.models import KerasModel

    wrap = KerasModel(model, (0, 255))

    attack = CarliniWagnerL2Attack(wrap)

    result = attack.as_generator(x)

    result = list(result)

    return result

def add_gaussian_noise(x, eps):
    result = np.copy(x)

    for i in range(len(result)):
        result[i] += np.random.normal(loc=0, scale=eps, size=result[i].shape)

    return np.clip(result, 0, 1.0)


def save_adversarials(path, images):
    np.save(os.path.join(path, "data.npy"), images)


def load_adversarials(path):
    return np.load(os.path.join(path, "data.npy"))
