import os

import PIL.Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.optimize import fmin_l_bfgs_b
from PIL import Image
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, save_img
from tensorflow.python.framework.ops import disable_eager_execution
import warnings

random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO: Take something cooler
CONTENT_IMG_PATH = "inputImage1.jpg"
STYLE_IMG_PATH = "styleImage1.jpg"

CONTENT_IMG_H = 100
CONTENT_IMG_W = 100

STYLE_IMG_H = 100
STYLE_IMG_W = 100

CONTENT_WEIGHT = 0.75  # Alpha weight.
STYLE_WEIGHT = 0.25  # Beta weight.
TOTAL_WEIGHT = 1e-60

TRANSFER_ROUNDS = 2

# =============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''


# https://keras.io/examples/generative/neural_style_transfer/
def deprocessImage(img):
    img = img.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


# ========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    numFilters = 3
    return K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (
            4. * (numFilters ^ 2) * ((CONTENT_IMG_H * CONTENT_IMG_W) ^ 2))


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


# https://keras.io/examples/generative/neural_style_transfer/
def totalLoss(x):
    a = tf.square(
        x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] -
        x[:, 1:, : CONTENT_IMG_W - 1, :]
    )
    b = tf.square(
        x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] -
        x[:, : CONTENT_IMG_H - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


def kAndFlatten(func):
    def F(x):
        x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
        ret = func([x])
        loss = ret[0]
        grad = ret[1].flatten().astype('float64')
        return loss, grad

    return F


# =========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return (
        (cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))


def preprocessData(raw):
    img, ih, iw = raw
    img = np.array(img.resize(size=(ih, iw)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


'''
TODO: A lot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''


def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))

    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])

    print("   VGG19 model loaded.")
    loss = K.variable(0.)
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"

    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    loss = loss + CONTENT_WEIGHT * contentLoss(contentOutput, genOutput)

    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        layer_f = outputDict[layerName]
        style_f = layer_f[1, :, :, :]
        out_f = layer_f[2, :, :, :]
        loss = loss + (STYLE_WEIGHT / len(styleLayerNames)) * styleLoss(style_f, out_f)
    print("   Calculating total var loss")
    loss = loss + TOTAL_WEIGHT * totalLoss(genTensor)

    print("   Setting up Gradients")
    grads = K.gradients(loss, genTensor)[0]
    outputs = [loss, grads]

    k_f = kAndFlatten(K.function([genTensor], outputs))  # Function that reshapes array to flat
    gen = tData.flatten()  # Start with input image

    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        gen_new, gen_loss, _ = fmin_l_bfgs_b(func=k_f, x0=gen, maxiter=20)

        gen = np.copy(gen_new)
        print("      Loss: ", gen_loss)

        img = deprocessImage(gen)
        saveFile = f"output_{i}.jpg"

        save_img(saveFile, img)
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")


# =========================<Main>================================================

def main():
    disable_eager_execution()  # K.gradients is not supported unless disabled
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])  # Content image.
    sData = preprocessData(raw[1])  # Style image.
    tData = preprocessData(raw[2])  # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")


if __name__ == "__main__":
    main()
