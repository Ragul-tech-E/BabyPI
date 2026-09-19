#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiUdp.h>
#include <ESPmDNS.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include "esp32-hal-ledc.h"

// ======================================================
// WIFI
// ======================================================

const char* WIFI_SSID = "bca";
const char* WIFI_PASSWORD = "12345678";

// ======================================================
// RASPBERRY PI DISCOVERY
// ======================================================

const int DISCOVERY_PORT = 4210;

// Raspberry Pi Flask server
const int PI_PORT = 5000;

// ======================================================
// TEMPERATURE SENSOR - DS18B20
// ======================================================

const int TEMP_PIN = 4;

OneWire oneWire(TEMP_PIN);
DallasTemperature sensors(&oneWire);

// ======================================================
// PWM
// ======================================================

const int pwmPin = 18;

const int pwmResolution = 10;
const int pwmFreq = 15625;

int pwmValue = 0;

// ======================================================
// UDP
// ======================================================

WiFiUDP udp;

// ======================================================
// TIMERS
// ======================================================

unsigned long lastTemperatureSend = 0;

const unsigned long temperatureInterval = 2000;

// ======================================================
// RASPBERRY PI IP
// ======================================================

IPAddress raspberryPiIP;

bool piFound = false;

// ======================================================
// WIFI CONNECT
// ======================================================

void connectWiFi()
{
  Serial.println();
  Serial.println("================================");
  Serial.println("Connecting to WiFi");
  Serial.println("================================");

  WiFi.mode(WIFI_STA);

  WiFi.setSleep(false);

  WiFi.begin(
    WIFI_SSID,
    WIFI_PASSWORD
  );

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi connected");

  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  Serial.print("Gateway: ");
  Serial.println(WiFi.gatewayIP());

  Serial.print("Subnet: ");
  Serial.println(WiFi.subnetMask());

  Serial.print("RSSI: ");
  Serial.print(WiFi.RSSI());
  Serial.println(" dBm");

  // ====================================================
  // START UDP
  // ====================================================

  if (udp.begin(DISCOVERY_PORT))
  {
    Serial.print("UDP started on port: ");
    Serial.println(DISCOVERY_PORT);
  }
  else
  {
    Serial.println("ERROR: UDP start failed");
  }
}

// ======================================================
// UDP DISCOVERY
// ======================================================

void handleDiscovery()
{
  int packetSize = udp.parsePacket();

  if (packetSize <= 0)
  {
    return;
  }

  char packet[100];

  int len = udp.read(
    packet,
    sizeof(packet) - 1
  );

  if (len <= 0)
  {
    return;
  }

  packet[len] = '\0';

  Serial.println();
  Serial.println("================================");
  Serial.println("UDP DISCOVERY RECEIVED");
  Serial.println("================================");

  Serial.print("Message: ");
  Serial.println(packet);

  Serial.print("Sender IP: ");
  Serial.println(udp.remoteIP());

  Serial.print("Sender Port: ");
  Serial.println(udp.remotePort());

  // ====================================================
  // RASPBERRY PI DISCOVER MESSAGE
  // ====================================================

  if (String(packet) == "ESP32_DISCOVER")
  {
    // --------------------------------------------------
    // SAVE RASPBERRY PI IP
    // --------------------------------------------------

    raspberryPiIP = udp.remoteIP();

    piFound = true;

    Serial.print("Raspberry Pi discovered: ");
    Serial.println(raspberryPiIP);

    // --------------------------------------------------
    // SEND RESPONSE
    // --------------------------------------------------

    String response = "ESP32_TEMP";

    udp.beginPacket(
      udp.remoteIP(),
      udp.remotePort()
    );

    udp.print(response);

    udp.endPacket();

    Serial.println("Discovery response sent");

    Serial.print("Pi IP saved as: ");
    Serial.println(raspberryPiIP);

    Serial.println("================================");
  }
}

// ======================================================
// SEND TEMPERATURE TO RASPBERRY PI
// ======================================================

void sendTemperature(float temperature)
{
  // ====================================================
  // CHECK WIFI
  // ====================================================

  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.println("ERROR: WiFi disconnected");
    return;
  }

  // ====================================================
  // CHECK PI
  // ====================================================

  if (!piFound)
  {
    Serial.println(
      "Raspberry Pi IP not known. Waiting for discovery..."
    );

    return;
  }

  // ====================================================
  // CREATE URL
  // ====================================================

  String url =
      "http://" +
      raspberryPiIP.toString() +
      ":" +
      String(PI_PORT) +
      "/temperature";

  Serial.println();
  Serial.println("================================");
  Serial.println("SEND TEMPERATURE");
  Serial.println("================================");

  Serial.print("Pi IP: ");
  Serial.println(raspberryPiIP);

  Serial.print("URL: ");
  Serial.println(url);

  Serial.print("Temperature: ");
  Serial.print(temperature, 2);
  Serial.println(" °C");

  // ====================================================
  // CREATE HTTP CLIENT
  // ====================================================

  HTTPClient http;

  http.setTimeout(3000);

  if (!http.begin(url))
  {
    Serial.println("ERROR: HTTP begin failed");

    return;
  }

  // ====================================================
  // HTTP HEADER
  // ====================================================

  http.addHeader(
    "Content-Type",
    "application/json"
  );

  // ====================================================
  // JSON
  // ====================================================

  String json = "{";

  json += "\"device\":\"esp32\",";
  json += "\"temperature\":";
  json += String(temperature, 2);

  json += "}";

  Serial.print("JSON: ");
  Serial.println(json);

  // ====================================================
  // POST
  // ====================================================

  int httpCode = http.POST(json);

  // ====================================================
  // RESPONSE
  // ====================================================

  if (httpCode > 0)
  {
    Serial.print("HTTP code: ");
    Serial.println(httpCode);

    String response = http.getString();

    Serial.print("Pi response: ");
    Serial.println(response);
  }
  else
  {
    Serial.print("HTTP error: ");
    Serial.println(
      http.errorToString(httpCode)
    );

    // --------------------------------------------------
    // Raspberry Pi may have changed IP
    // --------------------------------------------------

    piFound = false;

    Serial.println(
      "Pi marked as NOT FOUND."
    );

    Serial.println(
      "Waiting for new discovery..."
    );
  }

  http.end();

  Serial.println("================================");
}

// ======================================================
// SERIAL PWM COMMAND
// ======================================================

void handlePWM()
{
  if (!Serial.available())
  {
    return;
  }

  String input =
      Serial.readStringUntil('\n');

  input.trim();

  if (input.length() == 0)
  {
    return;
  }

  // ====================================================
  // CONVERT TO DUTY %
  // ====================================================

  double duty =
      input.toFloat();

  // ====================================================
  // LIMIT 0 - 100
  // ====================================================

  if (duty < 0)
  {
    duty = 0;
  }

  if (duty > 100.0)
  {
    duty = 100.0;
  }

  // ====================================================
  // CONVERT TO 10-BIT PWM
  // ====================================================

  pwmValue =
      (int)round(
        (duty / 100.0) * 1023.0
      );

  // ====================================================
  // WRITE PWM
  // ====================================================

  ledcWrite(
    pwmPin,
    pwmValue
  );

  // ====================================================
  // PRINT
  // ====================================================

  Serial.print("Duty = ");
  Serial.print(duty, 2);

  Serial.print("% -> PWM = ");
  Serial.println(pwmValue);
}

// ======================================================
// READ TEMPERATURE
// ======================================================

void readAndSendTemperature()
{
  sensors.requestTemperatures();

  float temperature =
      sensors.getTempCByIndex(0);

  // ====================================================
  // CHECK SENSOR
  // ====================================================

  if (
    temperature == DEVICE_DISCONNECTED_C ||
    temperature < -55.0 ||
    temperature > 125.0
  )
  {
    Serial.println();
    Serial.println(
      "ERROR: DS18B20 disconnected or invalid"
    );

    return;
  }

  // ====================================================
  // DISPLAY
  // ====================================================

  Serial.println();
  Serial.println("-------------------------------");

  Serial.print(
    "Temperature: "
  );

  Serial.print(
    temperature,
    2
  );

  Serial.println(" °C");

  Serial.println("-------------------------------");

  // ====================================================
  // SEND TO PI
  // ====================================================

  if (piFound)
  {
    sendTemperature(
      temperature
    );
  }
  else
  {
    Serial.println(
      "Pi IP not discovered yet."
    );
  }
}

// ======================================================
// SETUP
// ======================================================

void setup()
{
  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println();
  Serial.println("========================================");
  Serial.println("ESP32 TEMPERATURE + PWM");
  Serial.println("========================================");

  // ====================================================
  // PWM
  // ====================================================

  Serial.println();
  Serial.println("Initializing PWM...");

  ledcAttach(
    pwmPin,
    pwmFreq,
    pwmResolution
  );

  ledcWrite(
    pwmPin,
    0
  );

  Serial.print("PWM Pin: ");
  Serial.println(pwmPin);

  Serial.print("PWM Frequency: ");
  Serial.println(pwmFreq);

  Serial.print("PWM Resolution: ");
  Serial.println(pwmResolution);

  Serial.println("PWM initialized");

  // ====================================================
  // DS18B20
  // ====================================================

  Serial.println();
  Serial.println("Initializing DS18B20...");

  sensors.begin();

  int deviceCount =
      sensors.getDeviceCount();

  Serial.print(
    "DS18B20 sensors found: "
  );

  Serial.println(deviceCount);

  if (deviceCount == 0)
  {
    Serial.println(
      "WARNING: No DS18B20 detected!"
    );
  }

  // ====================================================
  // WIFI
  // ====================================================

  connectWiFi();

  // ====================================================
  // ESP32 mDNS
  // ====================================================

  Serial.println();
  Serial.println("Starting mDNS...");

  if (MDNS.begin("baby-esp32"))
  {
    Serial.println(
      "ESP32 mDNS started"
    );

    Serial.println(
      "Hostname: baby-esp32.local"
    );
  }
  else
  {
    Serial.println(
      "ERROR: mDNS failed"
    );
  }

  // ====================================================
  // READY
  // ====================================================

  Serial.println();
  Serial.println("========================================");
  Serial.println("ESP32 READY");
  Serial.println("========================================");

  Serial.println(
    "Waiting for Raspberry Pi discovery..."
  );

  Serial.println(
    "Send PWM duty through Serial:"
  );

  Serial.println(
    "Example: 50"
  );

  Serial.println(
    "Example: 75.5"
  );

  Serial.println();
}

// ======================================================
// LOOP
// ======================================================

void loop()
{
  // ====================================================
  // UDP DISCOVERY
  // ====================================================

  handleDiscovery();

  // ====================================================
  // PWM SERIAL COMMAND
  // ====================================================

  handlePWM();

  // ====================================================
  // TEMPERATURE
  // ====================================================

  if (
    millis() - lastTemperatureSend >=
    temperatureInterval
  )
  {
    lastTemperatureSend = millis();

    readAndSendTemperature();
  }

  // ====================================================
  // WIFI RECONNECT
  // ====================================================

  if (
    WiFi.status() != WL_CONNECTED
  )
  {
    Serial.println();
    Serial.println(
      "WARNING: WiFi lost!"
    );

    piFound = false;

    WiFi.disconnect();

    WiFi.begin(
      WIFI_SSID,
      WIFI_PASSWORD
    );

    Serial.println(
      "Attempting WiFi reconnect..."
    );

    delay(5000);

    if (WiFi.status() == WL_CONNECTED)
    {
      Serial.println(
        "WiFi reconnected!"
      );

      Serial.print(
        "New ESP32 IP: "
      );

      Serial.println(
        WiFi.localIP()
      );

      // Restart UDP
      udp.stop();

      udp.begin(
        DISCOVERY_PORT
      );

      Serial.println(
        "UDP discovery restarted"
      );
    }
  }

  // Small delay to keep loop responsive
  delay(5);
}