#include <Arduino.h>
#include <DHT.h>
#include <WiFi.h>
#include <ThingSpeak.h>

#include <Chirale_TensorFlowLite.h>
#include "model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------------- DHT11 ----------------

#define DHTPIN 4
#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);

// ---------------- WiFi ----------------

const char* WIFI_SSID     = "****";
const char* WIFI_PASSWORD = "***";

WiFiClient client;

// ---------------- ThingSpeak ----------------

unsigned long THINGSPEAK_CHANNEL_ID = ***;              // your channel ID
const char* THINGSPEAK_WRITE_API_KEY = "***";  // your Write API Key

// Field mapping (matches your ThingSpeak channel):
//   Field 1 -> current temperature reading
//   Field 2 -> model prediction
//   Field 3 -> humidity
#define TS_FIELD_TEMPERATURE 1
#define TS_FIELD_PREDICTION  2
#define TS_FIELD_HUMIDITY    3

// ---------------- TensorFlow ----------------

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 10 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// ---------------- Temperature Window ----------------

float temp_window[5];
int sample_count = 0;

// ---------------- WiFi helper ----------------

void connectWiFi() {
  Serial.print("Connecting to WiFi");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi connected, IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi connection failed, will retry in loop()");
  }
}

void ensureWiFiConnected() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }
}

void setup() {

  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("TinyML Temperature Prediction");

  dht.begin();

  connectWiFi();
  ThingSpeak.begin(client);

  // Load model
  model = tflite::GetModel(temperature_model_int8_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded successfully!");

  Serial.println();
  Serial.println("===== INPUT TENSOR =====");

  Serial.print("Type: ");
  Serial.println(input->type);

  Serial.print("Dimensions: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  Serial.print("Scale: ");
  Serial.println(input->params.scale, 8);

  Serial.print("Zero Point: ");
  Serial.println(input->params.zero_point);

  Serial.println();

  Serial.println("===== OUTPUT TENSOR =====");

  Serial.print("Type: ");
  Serial.println(output->type);

  Serial.print("Dimensions: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  Serial.print("Scale: ");
  Serial.println(output->params.scale, 8);

  Serial.print("Zero Point: ");
  Serial.println(output->params.zero_point);

  Serial.println("------------------------");
}

void loop() {

  ensureWiFiConnected();

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("DHT Error");
    delay(2000);
    return;
  }

  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" C");

  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.println(" %");

  // Shift window
  for (int i = 0; i < 4; i++)
    temp_window[i] = temp_window[i + 1];

  temp_window[4] = temperature;

  if (sample_count < 5)
    sample_count++;

  if (sample_count == 5) {

    Serial.println("Input values:");

    for (int i = 0; i < 5; i++) {

      int8_t q =
          (int8_t)round(temp_window[i] / input->params.scale +
                        input->params.zero_point);

      input->data.int8[i] = q;

      Serial.print(i);
      Serial.print(": ");
      Serial.print(temp_window[i]);
      Serial.print(" -> ");
      Serial.println(q);
    }

    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Inference Failed");
      return;
    }

    int8_t raw = output->data.int8[0];

    float prediction =
        (raw - output->params.zero_point) *
        output->params.scale;

    Serial.print("Raw Output: ");
    Serial.println(raw);

    Serial.print("Prediction: ");
    Serial.print(prediction);
    Serial.println(" C");

    // ---------------- ThingSpeak upload ----------------

    if (WiFi.status() == WL_CONNECTED) {

      ThingSpeak.setField(TS_FIELD_TEMPERATURE, temperature);
      ThingSpeak.setField(TS_FIELD_PREDICTION, prediction);
      ThingSpeak.setField(TS_FIELD_HUMIDITY, humidity);

      int httpCode = ThingSpeak.writeFields(THINGSPEAK_CHANNEL_ID, THINGSPEAK_WRITE_API_KEY);

      if (httpCode == 200) {
        Serial.println("ThingSpeak update successful");
      } else {
        Serial.print("ThingSpeak update failed, HTTP code: ");
        Serial.println(httpCode);
      }
    } else {
      Serial.println("WiFi not connected, skipping ThingSpeak upload");
    }

    Serial.println("------------------------");
  }

  // ThingSpeak free tier requires at least ~15s between updates from
  // the same channel; 60s loop delay already satisfies this.
  delay(20000);
}
