import os
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from fool import Fool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mpl.rcParams['figure.figsize'] = (5, 5)
mpl.rcParams['axes.grid'] = False
thing_index = 208  # we know it is a labrador retriever

def load_image(path='assets/test_image.jpg'):
    image_raw = tf.io.read_file(path)
    image = tf.image.decode_image(image_raw)
    return image


def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image


def get_imagenet_label(probs):
    return tf.keras.applications.mobilenet_v2.decode_predictions(probs, top=1)[0][0]


if __name__ == "__main__":

    # This loads and saves the model
    model = tf.keras.applications.MobileNetV2(
        include_top=True, weights='imagenet')
    model.trainable = False
    model.save('assets/model.h5')

    # This loads the image and classify it
    image = load_image()
    image = preprocess(image)
    probs = model.predict(image)

    # Something's wrong with the signal
    fool = Fool(thing_index, image, probs, model)
    hack = fool.predict()
    # fake_pic = fool._get_fake_pic()
    # noise = fool._make_noise()
    # plt.imshow(noise[0] * 0.5 + 0.5)

    # This plots the results
    plt.figure()
    plt.imshow(image[0]*0.5+0.5)
    _, image_class, class_confidence = get_imagenet_label(hack)
    result = '{} : {:.2f}% Confidence'.format(image_class, class_confidence*100)
    print(result)
    plt.title(result)
    plt.show()