

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("temperature_data.csv")

print(df.head())
print(df.shape)

window = 5

X = []
y = []

temps = df["temperature"].values

for i in range(len(temps) - window):
    X.append(temps[i:i+window])
    y.append(temps[i+window])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("First input:", X[0])
print("First target:", y[0])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, activation='relu', input_shape=(5,)),
    Dense(8, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

loss, mae = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Mean Absolute Error:", mae)

prediction = model.predict(X_test[:1])

print("Input:", X_test[0])
print("Predicted:", prediction[0][0])
print("Actual:", y_test[0])

def representative_dataset():
    for i in range(len(X_train)):
        yield [X_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

sample = X_test[3]

print("Input:", sample)

keras_pred = model.predict(np.expand_dims(sample, axis=0))
print("Keras:", keras_pred)

# Quantize input
input_scale = input_details[0]["quantization"][0]
input_zero = input_details[0]["quantization"][1]

sample_q = np.round(sample / input_scale + input_zero).astype(np.int8)

interpreter.set_tensor(
    input_details[0]["index"],
    np.expand_dims(sample_q, axis=0)
)

interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])

raw = int(output[0][0])

output_scale = output_details[0]["quantization"][0]
output_zero = output_details[0]["quantization"][1]

tflite_pred = (raw - output_zero) * output_scale

print("TFLite:", tflite_pred)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

print("Input scale:", input_details[0]["quantization"][0])
print("Input zero point:", input_details[0]["quantization"][1])

print("Output scale:", output_details[0]["quantization"][0])
print("Output zero point:", output_details[0]["quantization"][1])

sample = X_test[0]

print("Input:", sample)
print("Expected:", y_test[0])

scale = input_details[0]["quantization"][0]
zero = input_details[0]["quantization"][1]

sample_int8 = np.round(sample / scale + zero).astype(np.int8)

print(sample_int8)

interpreter.set_tensor(
    input_details[0]["index"],
    np.expand_dims(sample_int8, axis=0)
)

interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])

print(output)

output_scale = output_details[0]["quantization"][0]
output_zero = output_details[0]["quantization"][1]

prediction = (output[0][0] - output_zero) * output_scale

print("Prediction:", prediction)
print("Expected:", y_test[0])

interpreter = tf.lite.Interpreter(model_path="temperature_model_int8.tflite")
interpreter.allocate_tensors()

print(interpreter.get_input_details())
print(interpreter.get_output_details())

!xxd -i temperature_model_int8.tflite > model.h

import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="temperature_model_int8.tflite")
interpreter.allocate_tensors()

print("INPUT:")
print(interpreter.get_input_details())

print("\nOUTPUT:")
print(interpreter.get_output_details())

keras_prediction = model.predict(X_test[:10])

for i in range(10):
    print(
        "Input:", X_test[i],
        "Expected:", y_test[i],
        "Predicted:", keras_prediction[i][0]
    )