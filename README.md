# TinyML-Based Temperature Prediction on ESP32 using TensorFlow Lite Micro

A TinyML project that predicts the next temperature value directly on an ESP32 using a TensorFlow Lite Micro model trained on DHT11 sensor data. The project demonstrates Edge AI by performing machine learning inference on the microcontroller without requiring cloud computation.

---

## Features

- Temperature sensing using DHT11
- TinyML inference on ESP32
- TensorFlow Lite Micro deployment
- Sliding window temperature prediction
- Optional ThingSpeak cloud integration
- Low-memory INT8 quantized model

---

## Hardware

- ESP32 Dev Module
- DHT11 Temperature Sensor
- Breadboard
- Jumper Wires

---

## Software

- Arduino IDE
- Python
- TensorFlow / Keras
- TensorFlow Lite
- TensorFlow Lite Micro
- Google Colab

---

## Project Structure

```
ESP32-TinyML-Temperature-Prediction/
│
├── Arduino/
│   └── tinyml_temp_prediction.ino
│
├── Python/
│   └── train_model.ipynb
│
├── Model/
│   ├── temperature_model_int8.tflite
│   └── model.h
│
├── Images/
│
└── README.md
```

---

## Implementation


### ESP32 Setup

<img width="848" height="951" alt="image" src="https://github.com/user-attachments/assets/8ff3fbf1-8cd2-4aaa-bd68-f69363ac38ad" />


### Serial Monitor Output

<img width="511" height="669" alt="image" src="https://github.com/user-attachments/assets/6f148744-3bce-4343-acef-c2d4af8e8c1b" />


### ThingSpeak Dashboard (Optional)

<img width="634" height="421" alt="image" src="https://github.com/user-attachments/assets/31691f90-7eec-4a71-8736-64b9b9719ff7" />



