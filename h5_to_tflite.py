import tensorflow as tf
from tensorflow import keras

# Load the model architecture from the model.json file
with open('model.json', 'r') as json_file:
    model_json = json_file.read()

model = keras.models.model_from_json(model_json)
model.load_weights('model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open('model.tflite', 'wb') as tflite_file:
    tflite_file.write(tflite_model)
