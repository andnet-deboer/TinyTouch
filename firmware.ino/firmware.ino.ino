/*
  eFlesh dual-board sketch — 2x eFlesh (5 MLX90393 each) behind a PCA9546A I2C mux
  Target: Adafruit QT Py ESP32-S3, STEMMA QT cable to mux.

  Modes (select one at compile time):
    MODE_SERIAL   — binary framed protocol for collect.py / tinytouch.py (default)
    MODE_MICROROS — publishes raw sensor data as ROS2 Float32MultiArray on /tinytouch/raw
                    Requires: micro_ros_arduino library (Arduino Library Manager)
                    Run on host: docker run -it --rm -v /dev:/dev --privileged --net=host \
                                   microros/micro-ros-agent:jazzy serial --dev /dev/ttyACM0 -b 115200

  Based on the original eFlesh sketch by Venkatesh P (Beerware).
*/

// ── Mode selection ────────────────────────────────────────────────────────────
// Uncomment exactly one:
#define MODE_SERIAL
// #define MODE_MICROROS

// Node ID suffix for the ROS topic/node name — change per sensor unit
#ifndef NODE_ID
#define NODE_ID "0"
#endif

// ── Includes ──────────────────────────────────────────────────────────────────
#include <Wire.h>
#include <MLX90393.h>

#ifdef MODE_MICROROS
#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/float32_multi_array.h>
#endif

// ── PCA9546A ──────────────────────────────────────────────────────────────────
#define TCA_ADDR 0x70

static const uint8_t NUM_BOARDS = 2;
static const uint8_t BOARD_CHANNELS[NUM_BOARDS] = {0, 1};

// ── Sensors ───────────────────────────────────────────────────────────────────
static const uint8_t NUM_SENSORS_PER_BOARD = 5;
static const uint8_t NUM_SENSORS_TOTAL     = NUM_BOARDS * NUM_SENSORS_PER_BOARD;  // 10

MLX90393          mlx[NUM_BOARDS][NUM_SENSORS_PER_BOARD];
MLX90393::txyz    dataBuf[NUM_BOARDS][NUM_SENSORS_PER_BOARD];
uint8_t           initCount[NUM_BOARDS] = {0, 0};

const uint8_t TARGETS_ALL_CONSEC[NUM_SENSORS_PER_BOARD] = {0x0C, 0x0D, 0x0E, 0x0F, 0x10};
const uint8_t TARGETS_WHITE_SET[NUM_SENSORS_PER_BOARD]  = {0x0C, 0x10, 0x11, 0x12, 0x13};

// ── Forward declarations ──────────────────────────────────────────────────────
void tcaSelect(uint8_t channel);
void tcaDisableAll();
void scanI2C(uint8_t* found, uint8_t& count);
void chooseOrderedAddresses(const uint8_t* found, uint8_t count, uint8_t* ordered, uint8_t& orderedCount);
bool hasExactSet(const uint8_t* found, uint8_t count, const uint8_t* pattern);
void sortAscending(uint8_t* arr, uint8_t n);

// ─────────────────────────────────────────────────────────────────────────────
// microROS state (compiled out in MODE_SERIAL)
// ─────────────────────────────────────────────────────────────────────────────
#ifdef MODE_MICROROS

static rcl_publisher_t                  ros_pub;
static std_msgs__msg__Float32MultiArray ros_msg;
static rclc_support_t                   ros_support;
static rcl_allocator_t                  ros_allocator;
static rcl_node_t                       ros_node;
static rclc_executor_t                  ros_executor;
static bool                             ros_ready = false;

// Soft-fail: on RCL error, mark not-ready and return to allow reconnect attempts
#define RCCHECK(fn)     { if ((fn) != RCL_RET_OK) { ros_ready = false; return; } }
#define RCSOFTCHECK(fn) { (void)(fn); }

// Float32MultiArray layout: 10 sensors × 4 values = 40 floats
// Order per sensor: [t, x, y, z]  (boards 0-1, sensors 0-4 each)
static float ros_data[NUM_SENSORS_TOTAL * 4];

void ros_setup() {
    set_microros_transports();
    delay(2000);  // give the micro-ros-agent time to connect

    ros_allocator = rcl_get_default_allocator();
    RCCHECK(rclc_support_init(&ros_support, 0, NULL, &ros_allocator));
    RCCHECK(rclc_node_init_default(&ros_node, "tinytouch_" NODE_ID, "", &ros_support));
    RCCHECK(rclc_publisher_init_default(
        &ros_pub, &ros_node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
        "/tinytouch/raw"));

    ros_msg.data.data     = ros_data;
    ros_msg.data.size     = NUM_SENSORS_TOTAL * 4;
    ros_msg.data.capacity = NUM_SENSORS_TOTAL * 4;

    RCCHECK(rclc_executor_init(&ros_executor, &ros_support.context, 1, &ros_allocator));
    ros_ready = true;
}

void ros_publish() {
    if (!ros_ready) return;
    uint8_t k = 0;
    for (uint8_t b = 0; b < NUM_BOARDS; ++b)
        for (uint8_t i = 0; i < NUM_SENSORS_PER_BOARD; ++i) {
            ros_data[k++] = dataBuf[b][i].t;
            ros_data[k++] = dataBuf[b][i].x;
            ros_data[k++] = dataBuf[b][i].y;
            ros_data[k++] = dataBuf[b][i].z;
        }
    RCSOFTCHECK(rcl_publish(&ros_pub, &ros_msg, NULL));
    RCSOFTCHECK(rclc_executor_spin_some(&ros_executor, RCL_MS_TO_NS(0)));
}

#endif  // MODE_MICROROS

// ═════════════════════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 3000) { delay(5); }

    Wire.begin(41, 40);
    Wire.setClock(400000);
    Wire.setTimeOut(10);
    delay(10);

    tcaDisableAll();
    delay(2);

    Wire.beginTransmission(TCA_ADDR);
    if (Wire.endTransmission() != 0) {
        Serial.println(F("[ERROR] PCA9546A not found at 0x70."));
    } else {
        Serial.println(F("PCA9546A detected at 0x70."));
    }

    for (uint8_t b = 0; b < NUM_BOARDS; ++b) {
        Serial.print(F("\n=== Board ")); Serial.print(b);
        Serial.print(F(" on mux channel ")); Serial.println(BOARD_CHANNELS[b]);

        tcaSelect(BOARD_CHANNELS[b]);
        delay(2);

        uint8_t found[16] = {0}, foundCount = 0;
        scanI2C(found, foundCount);

        Serial.print(F("Found I2C addresses: "));
        for (uint8_t i = 0; i < foundCount; ++i) {
            Serial.print("0x"); Serial.print(found[i], HEX); Serial.print(' ');
        }
        Serial.println();

        uint8_t ordered[NUM_SENSORS_PER_BOARD] = {0}, orderedCount = 0;
        chooseOrderedAddresses(found, foundCount, ordered, orderedCount);

        if (orderedCount != NUM_SENSORS_PER_BOARD) {
            Serial.print(F("[WARN] Board ")); Serial.print(b);
            Serial.print(F(": found ")); Serial.print(orderedCount);
            Serial.println(F(" of 5 sensors."));
        }

        for (uint8_t i = 0; i < orderedCount; ++i) {
            byte status = mlx[b][i].begin(ordered[i], -1, Wire);
            Serial.print(F("Init MLX[")); Serial.print(b); Serial.print(F("][")); Serial.print(i);
            Serial.print(F("] @0x")); Serial.print(ordered[i], HEX);
            Serial.print(F(" status=0x")); Serial.println(status, HEX);
            mlx[b][i].setGainSel(0x1);
            mlx[b][i].setResolution(0x2, 0x2, 0x2);
            mlx[b][i].setDigitalFiltering(0x4);
            mlx[b][i].startBurst(0xF);
        }
        initCount[b] = orderedCount;
        for (uint8_t i = orderedCount; i < NUM_SENSORS_PER_BOARD; ++i)
            dataBuf[b][i] = {0, 0, 0, 0};
    }

    tcaDisableAll();

#ifdef MODE_SERIAL
    Serial.println(F("\n[MODE_SERIAL] Streaming binary frames."));
#endif
#ifdef MODE_MICROROS
    Serial.println(F("\n[MODE_MICROROS] Connecting to micro-ros-agent..."));
    ros_setup();
    if (ros_ready) Serial.println(F("microROS ready. Publishing on /tinytouch/raw"));
    else           Serial.println(F("[WARN] microROS agent not found — frames dropped until connected."));
#endif
}

// ═════════════════════════════════════════════════════════════════════════════
void loop() {
    // Read all sensors
    for (uint8_t b = 0; b < NUM_BOARDS; ++b) {
        tcaSelect(BOARD_CHANNELS[b]);
        for (uint8_t i = 0; i < initCount[b]; ++i)
            mlx[b][i].readBurstData(dataBuf[b][i]);
    }
    tcaDisableAll();

#ifdef MODE_SERIAL
    // Binary frame: 0xAA 0x55 header + 10 × sizeof(txyz) bytes
    const uint8_t header[2] = {0xAA, 0x55};
    Serial.write(header, 2);
    for (uint8_t b = 0; b < NUM_BOARDS; ++b)
        for (uint8_t i = 0; i < NUM_SENSORS_PER_BOARD; ++i)
            Serial.write((uint8_t*)&dataBuf[b][i], sizeof(dataBuf[b][i]));
    delayMicroseconds(500);
#endif

#ifdef MODE_MICROROS
    ros_publish();
    // ~50 Hz: 20 ms per loop. Sensor reads + RCL overhead fill most of this;
    // add a small guard delay to avoid spinning faster than the agent can consume.
    delay(10);
#endif
}

// ═════════════════════════════════════════════════════════════════════════════
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

void scanI2C(uint8_t* found, uint8_t& count) {
    count = 0;
    for (uint8_t addr = 0x08; addr <= 0x77; addr++) {
        if (addr == TCA_ADDR) continue;
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            found[count++] = addr;
            if (count >= 16) break;
        }
    }
}

void chooseOrderedAddresses(const uint8_t* found, uint8_t count,
                             uint8_t* ordered, uint8_t& orderedCount) {
    orderedCount = 0;
    if (count >= NUM_SENSORS_PER_BOARD) {
        if (hasExactSet(found, count, TARGETS_ALL_CONSEC)) {
            for (uint8_t i = 0; i < NUM_SENSORS_PER_BOARD; ++i) ordered[i] = TARGETS_ALL_CONSEC[i];
            orderedCount = NUM_SENSORS_PER_BOARD; return;
        }
        if (hasExactSet(found, count, TARGETS_WHITE_SET)) {
            ordered[0]=0x0C; ordered[1]=0x11; ordered[2]=0x12; ordered[3]=0x13; ordered[4]=0x10;
            orderedCount = NUM_SENSORS_PER_BOARD; return;
        }
    }
    uint8_t tmp[16], n = (count > 16) ? 16 : count;
    for (uint8_t i = 0; i < n; ++i) tmp[i] = found[i];
    sortAscending(tmp, n);
    orderedCount = (n >= NUM_SENSORS_PER_BOARD) ? NUM_SENSORS_PER_BOARD : n;
    for (uint8_t i = 0; i < orderedCount; ++i) ordered[i] = tmp[i];
}

bool hasExactSet(const uint8_t* found, uint8_t count, const uint8_t* pattern) {
    uint8_t hits = 0;
    for (uint8_t p = 0; p < NUM_SENSORS_PER_BOARD; ++p) {
        for (uint8_t i = 0; i < count; ++i)
            if (found[i] == pattern[p]) { ++hits; break; }
    }
    return hits == NUM_SENSORS_PER_BOARD;
}

void sortAscending(uint8_t* arr, uint8_t n) {
    for (uint8_t i = 0; i < n; ++i)
        for (uint8_t j = i+1; j < n; ++j)
            if (arr[j] < arr[i]) { uint8_t t=arr[i]; arr[i]=arr[j]; arr[j]=t; }
}
