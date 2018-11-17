####
# For an input batch of images, generates and adds one random (0 or 1) channel for each image.
####

import tensorflow as tf
import os
import numpy as np


def image2tensor():

    images_list = []
    for filename in os.listdir(os.path.join(os.getcwd(), "../fotos")):
        if filename.endswith(".jpg"):
            image = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image, channels=3)
            images_list.append(image_decoded)
            KNOWN_HEIGHT = 28
            KNOWN_WIDTH = 28
            image_decoded.set_shape([KNOWN_HEIGHT, KNOWN_WIDTH, 3])

    images_batch = tf.stack(images_list)
    print("batch of images: ",images_batch)
    return images_batch


def create_random_vector(images):
    size = images.get_shape()[0]
    random_vector = np.random.randint(2, size=size)
    print("random vector: ", random_vector)
    return random_vector

def add_channel(images):

    random_vector = create_random_vector(images)
    new_channels = np.array(random_vector) * np.array(tf.ones([28,28, 1], tf.uint8))

    #Així com per crear el vector és fàcil  passar-li el batch size i que te'l crei del tamany que toqui,
    #per passar larraylist a un tensor d'una dimensio més (és la unica manera que se mha acudit per multiplicar
    # vector i matriu al pas anterior, per això el resultat es un array list), no sé com ferho que no sigui de la
    #seguent manera si no es amb un for (i el dani m'ha demanat no fors jajajaj)
    new_channels = (tf.stack([tf.convert_to_tensor(new_channels[0]), tf.convert_to_tensor(new_channels[1]),
                    tf.convert_to_tensor(new_channels[2]), tf.convert_to_tensor(new_channels[3]),
                    tf.convert_to_tensor(new_channels[4])]))
    print("new channels: " , new_channels)

    increased_depth = tf.concat([images, new_channels], 3)
    print("original image with new channel added shape: ", increased_depth.get_shape())

    return increased_depth


if __name__ == '__main__':
    images_batch = image2tensor()
    returned_images = add_channel(images_batch)
