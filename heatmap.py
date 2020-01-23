import cv2
import numpy as np
import tensorflow as tf

# borrowed from: https://github.com/adityac94/Grad_CAM_plus_plus

def grad_CAM_plus(model, imgs):
    g = tf.get_default_graph()
    init = tf.global_variables_initializer()

    # Run tensorflow
    sess = tf.Session()

    # define your tensor placeholders for, labels and images
    label_vector = tf.compat.v1.placeholder("float32", [None, 43])
    label_index = tf.compat.v1.placeholder("int64", ())

    with tf.name_scope("content_vgg"):
        model.build(model.input)

    # get the output neuron corresponding to the class of interest (label_id)
    cost = model.output * label_vector

    # Get last convolutional layer gradients for generating gradCAM++ visualization
    target_conv_layer_output = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1].output
    target_conv_layer_grad = tf.gradients(cost, target_conv_layer_output)[0]

    # first_derivative
    first_derivative = tf.exp(cost)[0][label_index] * target_conv_layer_grad

    # second_derivative
    second_derivative = tf.exp(cost)[0][label_index] * target_conv_layer_grad * target_conv_layer_grad

    # triple_derivative
    triple_derivative = tf.exp(cost)[0][
                            label_index] * target_conv_layer_grad * target_conv_layer_grad * target_conv_layer_grad

    result = []

    sess.run(init)

    prob_vals = model.predict(imgs)

    for i, img in enumerate(imgs):
        output = [0.0] * model.output.get_shape().as_list()[1]  # one-hot embedding for desired class activations
        # creating the output vector for the respective class

        index = np.argmax(prob_vals[i])

        output[index] = 1.0
        label_id = index

        output = np.array(output)

        conv_output, conv_first_grad, conv_second_grad, conv_third_grad = sess.run(
            [target_conv_layer_output, first_derivative, second_derivative, triple_derivative],
            feed_dict={model.input: [img], label_index: label_id, label_vector: output.reshape((1, -1))})

        global_sum = np.sum(conv_output[0].reshape((-1, conv_first_grad[0].shape[2])), axis=0)

        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum.reshape(
            (1, 1, conv_first_grad[0].shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num / alpha_denom

        weights = np.maximum(conv_first_grad[0], 0.0)
        # normalizing the alphas
        """	
        alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)
    
        alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))
        """

        alphas_thresholding = np.where(weights, alphas, 0.0)

        alpha_normalization_constant = np.sum(np.sum(alphas_thresholding, axis=0), axis=0)
        alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant,
                                                          np.ones(alpha_normalization_constant.shape))

        alphas /= alpha_normalization_constant_processed.reshape((1, 1, conv_first_grad[0].shape[2]))

        deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad[0].shape[2])), axis=0)
        # print deep_linearization_weights
        grad_CAM_map = np.sum(deep_linearization_weights * conv_output[0], axis=2)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam)  # scale 0 to 1.0
        cam = cv2.resize(cam, (64, 64))

        cam = (cam * -1.0) + 1.0
        cam_heatmap = np.array(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET))

        cam_heatmap = cam_heatmap / 255.0

        fin = (img * 0.5) + (cam_heatmap * 0.5)
        result.append((fin, index))

    return np.array(result)
