/*
  eFlesh dual-board sketch — 2x eFlesh (5 MLX90393 each) behind a PCA9546A I2C mux
  Target: Adafruit QT Py ESP32-S3, STEMMA QT cable to mux.

  Based on the original eFlesh sketch by Venkatesh P (Beerware).
*/

#include <Wire.h>
#include <MLX90393.h>

// ---------- PCA9546A ----------
#define TCA_ADDR 0x70

static const uint8_t NUM_BOARDS = 2;
static const uint8_t BOARD_CHANNELS[NUM_BOARDS] = {0, 1};

// ---------- Sensors ----------
static const uint8_t NUM_SENSORS_PER_BOARD = 5;

MLX90393 mlx[NUM_BOARDS][NUM_SENSORS_PER_BOARD];
MLX90393::txyz dataBuf[NUM_BOARDS][NUM_SENSORS_PER_BOARD];

// Tracks how many sensors per board were successfully begin()'d.
// loop() only reads these — prevents null-deref crash when sensors are absent.
uint8_t initCount[NUM_BOARDS] = {0, 0};

const uint8_t TARGETS_ALL_CONSEC[NUM_SENSORS_PER_BOARD] = {0x0C, 0x0D, 0x0E, 0x0F, 0x10};
const uint8_t TARGETS_WHITE_SET[NUM_SENSORS_PER_BOARD]  = {0x0C, 0x10, 0x11, 0x12, 0x13};

// ---------- Forward decls ----------
void tcaSelect(uint8_t channel);
void tcaDisableAll();
void scanI2C(uint8_t* found, uint8_t& count);
void chooseOrderedAddresses(const uint8_t* found, uint8_t count, uint8_t* ordered, uint8_t& orderedCount);
bool hasExactSet(const uint8_t* found, uint8_t count, const uint8_t* pattern);
void sortAscending(uint8_t* arr, uint8_t n);

// =====================================================================
void setup() {
  Serial.begin(115200);
  // Don't block forever if USB-CDC isn't opened by a host:
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 3000) { delay(5); }

  Wire.begin(41, 40);
  Wire.setClock(400000);
  Wire.setTimeOut(10);  // ms — prevents I2C from hanging loop() if bus is stuck
  delay(10);

  // Make sure no channel is selected during the very first probe of the mux itself
  tcaDisableAll();
  delay(2);

  Wire.beginTransmission(TCA_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println(F("[ERROR] PCA9546A not found at 0x70. Check wiring/address straps."));
  } else {
    Serial.println(F("PCA9546A detected at 0x70."));
  }

  // Bring up each board on its own channel
  for (uint8_t b = 0; b < NUM_BOARDS; ++b) {
    Serial.print(F("\n=== Board ")); Serial.print(b);
    Serial.print(F(" on mux channel ")); Serial.print(BOARD_CHANNELS[b]);
    Serial.println(F(" ==="));

    tcaSelect(BOARD_CHANNELS[b]);
    delay(2);

    uint8_t found[16] = {0};
    uint8_t foundCount = 0;
    scanI2C(found, foundCount);

    Serial.print(F("Found I2C addresses: "));
    for (uint8_t i = 0; i < foundCount; ++i) {
      Serial.print("0x"); Serial.print(found[i], HEX); Serial.print(' ');
    }
    Serial.println();

    uint8_t ordered[NUM_SENSORS_PER_BOARD] = {0};
    uint8_t orderedCount = 0;
    chooseOrderedAddresses(found, foundCount, ordered, orderedCount);

    if (orderedCount != NUM_SENSORS_PER_BOARD) {
      Serial.print(F("[WARN] Board ")); Serial.print(b);
      Serial.print(F(": found ")); Serial.print(orderedCount);
      Serial.println(F(" of 5 expected MLX sensors."));
    }

    for (uint8_t i = 0; i < orderedCount; ++i) {
      byte status = mlx[b][i].begin(ordered[i], -1, Wire);
      Serial.print(F("Init MLX[")); Serial.print(b); Serial.print(F("][")); Serial.print(i);
      Serial.print(F("] @0x")); Serial.print(ordered[i], HEX);
      Serial.print(F(" status=0x")); Serial.println(status, HEX);

      mlx[b][i].setGainSel(0x1);
      mlx[b][i].setResolution(0x2, 0x2, 0x2);
      mlx[b][i].setDigitalFiltering(0x4);
      mlx[b][i].startBurst(0xF);  // T + X + Y + Z
    }

    initCount[b] = orderedCount;

    // Zero-fill slots that weren't initialised so frame size stays fixed
    for (uint8_t i = orderedCount; i < NUM_SENSORS_PER_BOARD; ++i) {
      dataBuf[b][i] = {0, 0, 0, 0};
    }
  }

  tcaDisableAll();
  Serial.println(F("\nStreaming. Board 0 sensors 0-4, then board 1 sensors 0-4."));
}

// =====================================================================
void loop() {
  // Switch to each board's channel, then read its 5 sensors before switching away.
  // Don't switch the mux between every sensor — wasted bus time.
  for (uint8_t b = 0; b < NUM_BOARDS; ++b) {
    tcaSelect(BOARD_CHANNELS[b]);
    for (uint8_t i = 0; i < initCount[b]; ++i) {
      mlx[b][i].readBurstData(dataBuf[b][i]);
    }
  }

  // Frame: 0xAA 0x55 header, then 10 × 16-byte sensor records
  const uint8_t header[2] = {0xAA, 0x55};
  Serial.write(header, 2);
  for (uint8_t b = 0; b < NUM_BOARDS; ++b) {
    for (uint8_t i = 0; i < NUM_SENSORS_PER_BOARD; ++i) {
      Serial.write((uint8_t*)&dataBuf[b][i], sizeof(dataBuf[b][i]));
    }
  }

  delayMicroseconds(500);
}

// =====================================================================
// PCA9546A helpers
void tcaSelect(uint8_t channel) {
  if (channel > 3) return;
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void tcaDisableAll() {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(0x00);
  Wire.endTransmission();
}

// =====================================================================
// (Below: unchanged from the original sketch.)
void scanI2C(uint8_t* found, uint8_t& count) {
  count = 0;
  for (uint8_t addr = 0x08; addr <= 0x77; addr++) {
    if (addr == TCA_ADDR) continue;  // the mux will answer too — skip it
    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();
    if (err == 0) {
      found[count++] = addr;
      if (count >= 16) break;
    }
  }
}

void chooseOrderedAddresses(const uint8_t* found, uint8_t count, uint8_t* ordered, uint8_t& orderedCount) {
  orderedCount = 0;

  if (count >= NUM_SENSORS_PER_BOARD) {
    if (hasExactSet(found, count, TARGETS_ALL_CONSEC)) {
      for (uint8_t i = 0; i < NUM_SENSORS_PER_BOARD; ++i) ordered[i] = TARGETS_ALL_CONSEC[i];
      orderedCount = NUM_SENSORS_PER_BOARD;
      return;
    }
    if (hasExactSet(found, count, TARGETS_WHITE_SET)) {
      ordered[0] = 0x0C; ordered[1] = 0x11; ordered[2] = 0x12; ordered[3] = 0x13; ordered[4] = 0x10;
      orderedCount = NUM_SENSORS_PER_BOARD;
      return;
    }
  }

  uint8_t tmp[16];
  uint8_t n = count;
  if (n > 16) n = 16;
  for (uint8_t i = 0; i < n; ++i) tmp[i] = found[i];
  sortAscending(tmp, n);

  orderedCount = (n >= NUM_SENSORS_PER_BOARD) ? NUM_SENSORS_PER_BOARD : n;
  for (uint8_t i = 0; i < orderedCount; ++i) ordered[i] = tmp[i];
}

bool hasExactSet(const uint8_t* found, uint8_t count, const uint8_t* pattern) {
  uint8_t hits = 0;
  for (uint8_t p = 0; p < NUM_SENSORS_PER_BOARD; ++p) {
    bool present = false;
    for (uint8_t i = 0; i < count; ++i) {
      if (found[i] == pattern[p]) { present = true; break; }
    }
    if (present) ++hits;
  }
  return (hits == NUM_SENSORS_PER_BOARD);
}

void sortAscending(uint8_t* arr, uint8_t n) {
  for (uint8_t i = 0; i < n; ++i) {
    for (uint8_t j = i + 1; j < n; ++j) {
      if (arr[j] < arr[i]) {
        uint8_t t = arr[i]; arr[i] = arr[j]; arr[j] = t;
      }
    }
  }
}