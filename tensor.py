import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs
from keras.models import load_model

# vgg16 = tf.keras.applications.vgg16.VGG16()

model__ = load_model("model.h5", compile=False)

model__.compile(
    loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

tfjs.converters.save_keras_model(model__,'tfjs/tfjs-models')