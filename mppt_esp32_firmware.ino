/*
 * =====================================================================
 * SOLARIS MPPT - ESP32 Firmware Implementation
 * On-Device 4-16-8-1 ANN Inference for 50W Solar Boost Converter
 * =====================================================================
 * Target: ESP32 (Xtensa Dual-Core 240MHz)
 * PWM Frequency: 10 kHz (Configured via ESP32 LEDC hardware timer)
 * Inputs: Irradiance (W/m²), Temperature (°C), V_pv (V), I_pv (A)
 * Output: Optimal Boost Converter Duty Cycle (D)
 * Execution Latency: ~2.5 microseconds per forward pass
 * Features:
 *   - Wi-Fi Station with SoftAP Fallback (SSID: Solaris-MPPT, IP: 192.168.4.1)
 *   - LittleFS Web Server for Embedded Dashboard
 *   - REST API (/api/predict, /api/status, /api/sensors) with CORS support
 * =====================================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <LittleFS.h>
#ifndef NN_WEIGHTS_H
#define NN_WEIGHTS_H

// =====================================================================
// Trained Neural Network Parameters (4-16-8-1 MPPT Architecture)
// Trained on 50W Solar Panel + Boost Converter Dataset
// Inputs: [Irradiance (W/m2), Temperature (C), V_pv (V), I_pv (A)]
// Output: [Optimal Duty Cycle (0.10 to 0.85)]
// =====================================================================

// 1. MinMax Scaler Parameters: x_norm = x * SCALER_SCALE + SCALER_MIN
const float SCALER_SCALE[4] = { 0.0011122f, 0.0222297f, 0.0358945f, 0.1259167f };
const float SCALER_MIN[4]   = { -0.1120437f, -0.3337604f, -0.3089309f, -0.0383882f };

// 2. Hidden Layer 1 (Dense: 4 -> 16, ReLU)
const float W1[4][16] = {
  { 0.3923765f, 0.5468810f, -0.1624799f, 0.5832646f, -1.5947528f, -0.2771439f, 0.4592175f, -0.2041521f, -0.0032284f, 0.1036578f, -0.4530561f, 0.1528988f, 0.7693197f, 0.1982577f, -0.0075787f, 0.6903867f },
  { -0.5270954f, 0.0289930f, 0.0041200f, -0.2227012f, -0.0121235f, 0.3246362f, 0.3669606f, -0.1346494f, 0.0233551f, -0.4374786f, -0.0413143f, 0.2112878f, 0.0189003f, -0.1706990f, -0.3263734f, 0.5829484f },
  { -0.4273409f, 0.0231089f, -0.2679682f, -0.3120934f, 0.0269524f, -0.3691865f, 0.1465634f, 0.2398841f, -0.5737793f, 0.0295474f, -0.1817106f, -0.2104064f, 0.5490125f, 0.0496116f, 0.0243322f, -0.0937853f },
  { -0.4131831f, 0.3475079f, 0.1182136f, -0.0773097f, -0.0151302f, -0.2020875f, 0.2108807f, 0.5278191f, -0.0463903f, -0.2357044f, -0.5115089f, -0.2419805f, 0.1871960f, 0.4089109f, -0.1460464f, 0.2092431f }
};
const float BIAS1[16] = { 0.0000000f, -0.2019813f, -0.0114317f, 0.2866067f, 0.0946232f, 0.1364733f, -0.0882008f, 0.0942474f, 0.3167799f, -0.0381091f, 0.0000000f, 0.4407181f, -0.0410895f, 0.1178465f, -0.0100793f, 0.0480995f };

// 3. Hidden Layer 2 (Dense: 16 -> 8, ReLU)
const float W2[16][8] = {
  { 0.4791559f, -0.0179821f, 0.2660979f, 0.2428727f, 0.3159993f, -0.3813212f, 0.2031455f, 0.0293988f },
  { -0.8298886f, -0.1903392f, -0.3124661f, -0.0206975f, 0.3643341f, 0.1398580f, -0.0403225f, -0.4963551f },
  { 0.2409482f, 0.3369271f, -0.3961801f, 0.0542548f, 0.2709185f, 0.4650519f, 0.0430354f, 0.4378563f },
  { 0.2207691f, 0.2957780f, -0.0832727f, -0.3406036f, 0.4257004f, 0.0028452f, -0.3060538f, -0.0075634f },
  { -0.7234812f, 0.4547365f, -0.3828110f, -0.4738039f, -0.4736143f, -0.3341761f, -0.4127770f, -0.1579327f },
  { -0.0411165f, 0.5255395f, -0.0729272f, 0.2150599f, -0.0372747f, 0.4512323f, -0.5494049f, -0.3313347f },
  { -0.1384151f, -0.2256503f, 0.1984802f, 0.2140978f, 0.4406615f, 0.0203958f, 0.2937562f, 0.2204344f },
  { 0.0504821f, 0.5008378f, 0.0136712f, -0.2820444f, -0.2882254f, -0.2900083f, -0.4528862f, -0.1458258f },
  { 0.1180177f, 0.1116881f, 0.0042824f, -0.2491876f, 0.5195464f, -0.2861404f, 0.4298366f, 0.0615244f },
  { -0.3602371f, -0.0616500f, 0.2050786f, 0.4303159f, 0.3959532f, -0.3232050f, -0.0274969f, -0.4971688f },
  { 0.1915498f, -0.0442172f, 0.3452333f, 0.3764613f, 0.3136108f, -0.1805680f, -0.2703552f, 0.1997812f },
  { 0.2352575f, -0.2604248f, -0.4316528f, 0.0069683f, -1.3603246f, 0.3437209f, -0.1757472f, -0.4703527f },
  { 0.3707612f, -0.1707773f, 0.0980729f, 0.2088214f, 0.2392405f, -0.2540730f, 0.0503757f, -0.1141578f },
  { 0.7128500f, -0.2626098f, -0.5204203f, -0.2944249f, -0.2204599f, -0.4520288f, -0.5314005f, -0.2577034f },
  { 0.0057472f, 0.4765564f, -0.5017651f, 0.2952313f, -0.2794077f, 0.1973978f, 0.0300614f, -0.2894574f },
  { 0.4160668f, -0.2932369f, -0.1461529f, -0.3165213f, 0.3976471f, -0.4940864f, -0.0181952f, 0.0734388f }
};
const float BIAS2[8]     = { 0.0920750f, 0.0354347f, -0.0556939f, -0.0608862f, -0.1823883f, -0.0316299f, -0.0565602f, -0.0516619f };

// 4. Output Layer (Dense: 8 -> 1, Linear)
const float W3[8][1]  = {
  { 0.8282046f },
  { -0.0529624f },
  { -0.1502742f },
  { -0.4699148f },
  { -0.1699545f },
  { 0.5824218f },
  { -0.6170049f },
  { -0.0579119f }
};
const float BIAS3[1]     = { 0.0857476f };

#endif // NN_WEIGHTS_H


// =====================================================================
// 1. HARDWARE PIN & PWM DEFINITIONS
// =====================================================================
#define PIN_PWM_GATE     18   // PWM output to MOSFET Gate Driver (TC4420 / IR2104)
#define PIN_ADC_VPV      34   // Solar Panel Voltage divider (0-22V mapped to 0-3.3V)
#define PIN_ADC_IPV      35   // Current Sensor (ACS712-05B / INA219 / Shunt)
#define PIN_ADC_TEMP     32   // Temperature Sensor (NTC Thermistor / LM35)
#define PIN_ADC_IRR      33   // Irradiance Sensor (Reference Photodiode)
#define PIN_LED_STATUS   2    // Built-in status LED

// Hardware PWM Parameters
#define PWM_FREQ         10000 // 10 kHz switching frequency
#define PWM_CHANNEL      0     // LEDC Channel 0
#define PWM_RESOLUTION   10    // 10-bit resolution (Duty cycle values: 0 to 1023)
#define PWM_MAX_VAL      1023  // (1 << PWM_RESOLUTION) - 1

// Safety Duty Cycle limits
#define MIN_DUTY         0.10f // 10% minimum duty cycle
#define MAX_DUTY         0.85f // 85% maximum duty cycle

// Core 2.x vs 3.x Compatibility for LEDC PWM
#if defined(ESP_ARDUINO_VERSION) && (ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0))
  #define INIT_PWM() ledcAttach(PIN_PWM_GATE, PWM_FREQ, PWM_RESOLUTION)
  #define WRITE_PWM(duty_val) ledcWrite(PIN_PWM_GATE, duty_val)
#else
  #define INIT_PWM() do { ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION); ledcAttachPin(PIN_PWM_GATE, PWM_CHANNEL); } while(0)
  #define WRITE_PWM(duty_val) ledcWrite(PWM_CHANNEL, duty_val)
#endif

// =====================================================================
// 2. WI-FI & WEBSERVER CONFIGURATION
// =====================================================================
// Customize with your Wi-Fi router credentials:
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

// SoftAP Fallback (Always available if router connection fails)
const char* AP_SSID = "Solaris-MPPT";
const char* AP_PASS = "12345678"; // Min 8 characters

WebServer server(80);

// Operating Mode: 0 = Hardware ADC loop, 1 = Web / Simulation controlled
enum OperatingMode { MODE_HARDWARE = 0, MODE_SIMULATION = 1 };
OperatingMode currentMode = MODE_SIMULATION;

// Live System State
struct SystemState {
  float irradiance = 800.0f;
  float temperature = 35.0f;
  float v_pv = 17.0f;
  float i_pv = 2.0f;
  float p_pv = 34.0f;
  float duty = 0.25f;
  uint32_t raw_pwm = 256;
  uint32_t latency_us = 3;
  uint32_t total_predictions = 0;
  bool is_ap_mode = false;
  String ip_address = "";
} state;

// =====================================================================
// 3. NEURAL NETWORK FORWARD PASS (4-16-8-1 ANN)
// =====================================================================
float run_ann_inference(float irr, float temp, float vpv, float ipv, uint32_t &latency_micros) {
  uint32_t t_start = micros();

  // Step 1: MinMax Normalization (x_norm = x * SCALER_SCALE + SCALER_MIN)
  float raw_in[4] = { irr, temp, vpv, ipv };
  float x_norm[4];
  for (int i = 0; i < 4; i++) {
    x_norm[i] = raw_in[i] * SCALER_SCALE[i] + SCALER_MIN[i];
  }

  // Step 2: Hidden Layer 1 (Dense: 4 -> 16, ReLU)
  float h1[16];
  for (int j = 0; j < 16; j++) {
    float sum = BIAS1[j];
    for (int i = 0; i < 4; i++) {
      sum += x_norm[i] * W1[i][j];
    }
    h1[j] = (sum > 0.0f) ? sum : 0.0f; // ReLU
  }

  // Step 3: Hidden Layer 2 (Dense: 16 -> 8, ReLU)
  float h2[8];
  for (int k = 0; k < 8; k++) {
    float sum = BIAS2[k];
    for (int j = 0; j < 16; j++) {
      sum += h1[j] * W2[j][k];
    }
    h2[k] = (sum > 0.0f) ? sum : 0.0f; // ReLU
  }

  // Step 4: Output Layer (Dense: 8 -> 1, Linear)
  float out_duty = BIAS3[0];
  for (int k = 0; k < 8; k++) {
    out_duty += h2[k] * W3[k][0];
  }

  // Step 5: Hardware Clamping (Safety protection for boost inductor)
  if (out_duty < MIN_DUTY) out_duty = MIN_DUTY;
  if (out_duty > MAX_DUTY) out_duty = MAX_DUTY;

  latency_micros = micros() - t_start;
  return out_duty;
}

void apply_duty_cycle(float duty) {
  state.duty = duty;
  state.raw_pwm = (uint32_t)(duty * (float)PWM_MAX_VAL);
  if (state.raw_pwm > PWM_MAX_VAL) state.raw_pwm = PWM_MAX_VAL;
  WRITE_PWM(state.raw_pwm);
}

// =====================================================================
// 4. REST API & HTTP SERVER HANDLERS
// =====================================================================
void send_cors_headers() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
}

void handle_options() {
  send_cors_headers();
  server.send(204);
}

void handle_api_status() {
  send_cors_headers();

  char json[384];
  snprintf(json, sizeof(json),
    "{\"status\":\"ok\",\"mode\":\"%s\",\"ip\":\"%s\",\"is_ap\":%s,"
    "\"free_heap\":%u,\"uptime_s\":%lu,\"predictions\":%u,"
    "\"duty\":%.4f,\"raw_pwm\":%u,\"latency_us\":%u}",
    (currentMode == MODE_SIMULATION) ? "SIMULATION" : "HARDWARE",
    state.ip_address.c_str(),
    state.is_ap_mode ? "true" : "false",
    ESP.getFreeHeap(),
    millis() / 1000UL,
    state.total_predictions,
    state.duty,
    state.raw_pwm,
    state.latency_us
  );

  server.send(200, "application/json", json);
}

void handle_api_predict() {
  send_cors_headers();

  float irr = state.irradiance;
  float temp = state.temperature;
  float vpv = state.v_pv;
  float ipv = state.i_pv;

  // Read query parameters: ?irr=...&temp=...&vpv=...&ipv=...
  if (server.hasArg("irr")) irr = server.arg("irr").toFloat();
  if (server.hasArg("temp")) temp = server.arg("temp").toFloat();
  if (server.hasArg("vpv")) vpv = server.arg("vpv").toFloat();
  if (server.hasArg("ipv")) ipv = server.arg("ipv").toFloat();

  // Or read JSON body if POST
  if (server.hasArg("plain")) {
    String body = server.arg("plain");
    int i_irr = body.indexOf("\"irr\"");
    if (i_irr > 0) irr = body.substring(body.indexOf(':', i_irr) + 1).toFloat();
    int i_temp = body.indexOf("\"temp\"");
    if (i_temp > 0) temp = body.substring(body.indexOf(':', i_temp) + 1).toFloat();
    int i_vpv = body.indexOf("\"vpv\"");
    if (i_vpv > 0) vpv = body.substring(body.indexOf(':', i_vpv) + 1).toFloat();
    int i_ipv = body.indexOf("\"ipv\"");
    if (i_ipv > 0) ipv = body.substring(body.indexOf(':', i_ipv) + 1).toFloat();
  }

  // Execute on-device neural forward pass
  uint32_t latency = 0;
  float pred_duty = run_ann_inference(irr, temp, vpv, ipv, latency);

  // Update physical PWM and internal state
  apply_duty_cycle(pred_duty);
  state.irradiance = irr;
  state.temperature = temp;
  state.v_pv = vpv;
  state.i_pv = ipv;
  state.p_pv = vpv * ipv;
  state.latency_us = latency;
  state.total_predictions++;

  // Calculate boost impedance matching metrics for 15 Ohm load
  float r_mpp = (ipv > 0.01f) ? (vpv / ipv) : 999.0f;
  float boost_vout = (pred_duty < 0.99f) ? (vpv / (1.0f - pred_duty)) : vpv;

  char json[512];
  snprintf(json, sizeof(json),
    "{\"status\":\"ok\",\"duty\":%.4f,\"duty_pct\":%.2f,\"raw_pwm\":%u,\"pwm_max\":%u,"
    "\"latency_us\":%u,\"inputs\":{\"irr\":%.1f,\"temp\":%.1f,\"vpv\":%.3f,\"ipv\":%.3f},"
    "\"calculated\":{\"p_pv\":%.2f,\"r_mpp\":%.2f,\"boost_vout\":%.2f},\"predictions\":%u}",
    pred_duty,
    pred_duty * 100.0f,
    state.raw_pwm,
    PWM_MAX_VAL,
    latency,
    irr, temp, vpv, ipv,
    state.p_pv,
    r_mpp,
    boost_vout,
    state.total_predictions
  );

  server.send(200, "application/json", json);
}

void handle_file_read() {
  String path = server.uri();
  if (path.endsWith("/")) path += "index.html";

  String contentType = "text/plain";
  if (path.endsWith(".html")) contentType = "text/html";
  else if (path.endsWith(".css")) contentType = "text/css";
  else if (path.endsWith(".js")) contentType = "application/javascript";
  else if (path.endsWith(".json")) contentType = "application/json";
  else if (path.endsWith(".ico")) contentType = "image/x-icon";

  if (LittleFS.exists(path)) {
    File file = LittleFS.open(path, "r");
    server.streamFile(file, contentType);
    file.close();
    return;
  }

  // Fallback response if index.html hasn't been uploaded to LittleFS yet
  String fallbackHtml = "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Solaris MPPT ESP32</title>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<style>body{font-family:sans-serif;background:#0d1117;color:#c9d1d9;padding:2rem;text-align:center;}"
    ".card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:2rem;max-width:500px;margin:2rem auto;}"
    "h1{color:#f59e0b;}button{background:#f59e0b;color:#000;border:none;padding:12px 24px;border-radius:6px;font-weight:bold;cursor:pointer;}"
    "</style></head><body><div class='card'>"
    "<h1>⚡ Solaris MPPT ESP32 Online</h1>"
    "<p>Neural Network: <strong>4-16-8-1 ANN Active</strong></p>"
    "<p>PWM Frequency: <strong>10 kHz (GPIO 18)</strong></p>"
    "<p>Current Duty: <strong>" + String(state.duty * 100.0f, 1) + "%</strong></p>"
    "<p>LittleFS Status: Upload <code>data/index.html</code> via PlatformIO / Arduino IDE to see the full UI.</p>"
    "<p><a href='/api/status' style='color:#38bdf8'>View /api/status</a> | <a href='/api/predict?irr=1000&temp=25&vpv=18.0&ipv=2.78' style='color:#38bdf8'>Test /api/predict</a></p>"
    "</div></body></html>";

  server.send(200, "text/html", fallbackHtml);
}

// =====================================================================
// 5. SETUP & MAIN LOOP
// =====================================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n==================================================");
  Serial.println("  SOLARIS MPPT - ESP32 4-16-8-1 NEURAL INFERENCE");
  Serial.println("==================================================");

  // Initialize Status LED
  pinMode(PIN_LED_STATUS, OUTPUT);
  digitalWrite(PIN_LED_STATUS, LOW);

  // Initialize Hardware PWM on GPIO 18
  INIT_PWM();
  apply_duty_cycle(0.10f); // Default safe start at 10% duty cycle
  Serial.printf("[PWM] Initialized on GPIO %d at %d Hz (10-bit)\n", PIN_PWM_GATE, PWM_FREQ);

  // Initialize LittleFS Flash Storage
  if (!LittleFS.begin(true)) {
    Serial.println("[LittleFS] Warning: LittleFS mount failed, will use fallback server");
  } else {
    Serial.println("[LittleFS] Mounted successfully!");
  }

  // Connect to Wi-Fi Station Mode
  WiFi.mode(WIFI_AP_STA);
  Serial.printf("[WiFi] Connecting to %s...\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  unsigned long startAttempt = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startAttempt < 5000) {
    digitalWrite(PIN_LED_STATUS, !digitalRead(PIN_LED_STATUS));
    delay(250);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    state.is_ap_mode = false;
    state.ip_address = WiFi.localIP().toString();
    Serial.printf("\n[WiFi] Connected! IP Address: http://%s\n", state.ip_address.c_str());
    digitalWrite(PIN_LED_STATUS, HIGH);
  } else {
    // Wi-Fi Station failed -> Start SoftAP so browser can ALWAYS connect!
    Serial.println("\n[WiFi] Station connection timed out. Starting SoftAP fallback...");
    WiFi.softAP(AP_SSID, AP_PASS);
    state.is_ap_mode = true;
    state.ip_address = WiFi.softAPIP().toString();
    Serial.printf("[WiFi] SoftAP Active! SSID: '%s', Password: '%s'\n", AP_SSID, AP_PASS);
    Serial.printf("[WiFi] Web Interface available at: http://%s\n", state.ip_address.c_str());
  }

  // Setup Web Server Routes
  server.on("/api/status", HTTP_GET, handle_api_status);
  server.on("/api/predict", HTTP_GET, handle_api_predict);
  server.on("/api/predict", HTTP_POST, handle_api_predict);
  server.on("/api/status", HTTP_OPTIONS, handle_options);
  server.on("/api/predict", HTTP_OPTIONS, handle_options);

  // Serve LittleFS files or fallback HTML for any other route
  server.onNotFound(handle_file_read);

  server.begin();
  Serial.println("[HTTP] Web Server started on port 80!");
  Serial.println("==================================================\n");

  // Initial test prediction at Standard Test Conditions (1000 W/m², 25°C, 18V, 2.78A)
  uint32_t t_us = 0;
  float test_d = run_ann_inference(1000.0f, 25.0f, 18.0f, 2.78f, t_us);
  apply_duty_cycle(test_d);
  Serial.printf("[ANN Test] STC Input -> Duty: %.4f (%.2f%%) in %u microseconds\n\n",
                test_d, test_d * 100.0f, t_us);
}

void loop() {
  server.handleClient();

  // In hardware mode, read physical ADCs periodically
  static unsigned long lastAdcRead = 0;
  if (currentMode == MODE_HARDWARE && millis() - lastAdcRead >= 50) {
    lastAdcRead = millis();
    // Read ADCs (ADC attenuation calibrated for 0-3.3V)
    uint16_t raw_v = analogRead(PIN_ADC_VPV);
    uint16_t raw_i = analogRead(PIN_ADC_IPV);
    uint16_t raw_t = analogRead(PIN_ADC_TEMP);
    uint16_t raw_g = analogRead(PIN_ADC_IRR);

    // Map to physical units
    state.v_pv = (raw_v / 4095.0f) * 22.0f;
    state.i_pv = (raw_i / 4095.0f) * 3.0f;
    state.temperature = 15.0f + (raw_t / 4095.0f) * 50.0f;
    state.irradiance = 100.0f + (raw_g / 4095.0f) * 900.0f;

    uint32_t lat = 0;
    float d = run_ann_inference(state.irradiance, state.temperature, state.v_pv, state.i_pv, lat);
    apply_duty_cycle(d);
  }
}
