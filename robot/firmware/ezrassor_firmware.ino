/*
 * robot_control.ino
 *
 * Two serial command protocols:
 *
 *  MODE A — 1-byte command (original accel-ramp logic, unchanged):
 *    0x00  STOP   – begin deceleration ramp to stop
 *    0x01  FWD    – increment forward speed tier (up to +3)
 *    0x02  REV    – increment reverse speed tier (up to -3)
 *    0x03  LEFT   – left side reverse, right side forward
 *    0x04  RIGHT  – right side reverse, left side forward
 *    0xFF  HALT   – immediate stop
 *    (any other byte also triggers a stop, matching original behavior)
 *
 *  MODE B — 4-byte packet (direction + raw step count):
 *    Byte 0: 0xA0          – header to distinguish from Mode A
 *    Byte 1: direction     – 0x01 FWD | 0x02 REV | 0x03 LEFT | 0x04 RIGHT
 *    Byte 2: steps high    – upper 8 bits of target step count
 *    Byte 3: steps low     – lower 8 bits of target step count
 *
 *    Total step range: 0–65535 motor steps.
 *    Speed is fixed at MODE_B_RPM.
 *    steps == 0 means run indefinitely at MODE_B_RPM until a Mode A stop arrives.
 *
 *  MICROSTEP_MODE 1 = no jumpers installed on RAMPS 1.4.
 */

#include "DRV8825.h"
#include "Adafruit_VL53L0X.h"

// ============ MICROSTEPPING ============
#define MICROSTEP_MODE   1
#define STEPS_PER_REV    (200 * MICROSTEP_MODE)
// =======================================

// ============ MODE A SPEED =============
#define MAX_SPEED        90
#define ACCEL_INTERVAL   100
#define ACCEL_INCREMENT  10
#define ACCEL_MAX        40
// =======================================

// ============ MODE B SPEED =============
#define MODE_B_RPM       100
// =======================================

// ============ SENSOR CONFIG ============
#define PRINT_SENSOR_DEBUG          0
#define LIDAR_INTERVAL_MS           100
#define ULTRASONIC_STEP_INTERVAL_MS 25
#define ULTRASONIC_TIMEOUT_US       12000UL
#define OBSTACLE_THRESHOLD_CM       30
// =======================================

// ---- RAMPS 1.4 pins -------------------
#define X_STEP_PIN    54
#define X_DIR_PIN     55
#define X_ENABLE_PIN  38

#define Y_STEP_PIN    60
#define Y_DIR_PIN     61
#define Y_ENABLE_PIN  56

#define Z_STEP_PIN    46
#define Z_DIR_PIN     48
#define Z_ENABLE_PIN  62

#define E0_STEP_PIN   26
#define E0_DIR_PIN    28
#define E0_ENABLE_PIN 24

#define E1_STEP_PIN   36
#define E1_DIR_PIN    34
#define E1_ENABLE_PIN 30

// ---- Sensors --------------------------
#define SHT_LOX    4
#define FRONT_TRIG 57
#define FRONT_ECHO 58
#define BL_TRIG    63
#define BL_ECHO    59
#define BR_TRIG    52
#define BR_ECHO    53

// ---- Motor objects --------------------
DRV8825 backRight (X_STEP_PIN,  X_DIR_PIN,  X_ENABLE_PIN,  STEPS_PER_REV);
DRV8825 frontRight(Y_STEP_PIN,  Y_DIR_PIN,  Y_ENABLE_PIN,  STEPS_PER_REV);
DRV8825 frontLeft (E1_STEP_PIN, E1_DIR_PIN, E1_ENABLE_PIN, STEPS_PER_REV);
DRV8825 backLeft  (E0_STEP_PIN, E0_DIR_PIN, E0_ENABLE_PIN, STEPS_PER_REV);

Adafruit_VL53L0X lox = Adafruit_VL53L0X();

// ============================================================
//  COMMAND BYTES
// ============================================================
enum Commands : byte {
    STOP          = 0x00,
    FWD           = 0x01,
    REV           = 0x02,
    LEFT          = 0x03,
    RIGHT         = 0x04,
    HALT          = 0xFF,
    FRONT         = 0x05,
    BACK          = 0x06,
    RAISE         = 0x07,
    LOWER         = 0x08,
    MODE_B_HEADER = 0xA0
};

// ============================================================
//  MODE B PACKET STATE MACHINE
// ============================================================
enum PacketState : byte {
    PKT_IDLE,
    PKT_BYTE1,
    PKT_BYTE2,
    PKT_BYTE3
};

PacketState packetState = PKT_IDLE;
byte        pktDir      = 0;
byte        pktHigh     = 0;

// ============================================================
//  MODE A STATE
// ============================================================
byte last_command = STOP;
unsigned long last_command_time = 0;
unsigned long command_timeout   = 10000000000UL;

long maxSpeed = MAX_SPEED;

long idle_left_speed  = 1;
long idle_right_speed = 1;

bool rs = false;
bool ls = false;

int on        = 0;
int stopFlag  = 0;
int slowSpeed = ACCEL_INCREMENT;
int leftSpeed  = 0;
int rightSpeed = 0;

unsigned long lastRampMs = 0;

// ============================================================
//  MODE B STATE
// ============================================================
bool modeBActive = false;
long targetSteps = 0;
long startPosFL  = 0;
long startPosBL  = 0;
long startPosFR  = 0;
long startPosBR  = 0;

// ============================================================
//  SENSOR STATE
// ============================================================
unsigned long lastLidarMs       = 0;
unsigned long lastUltrasonicMs  = 0;

uint16_t lidarDistanceMm = 0;
bool     lidarValid      = false;
float    frontDistCm     = -1.0f;
float    blDistCm        = -1.0f;
float    brDistCm        = -1.0f;
uint8_t  ultrasonicPhase = 0;

// ============================================================
//  FORWARD DECLARATIONS
// ============================================================
void update_motors();
void read_serial();
void handle_mode_a(byte cmd);
void handle_mode_b(byte dir, uint16_t steps);
void apply_direction_b(byte dir);
void set_all_speed(long rpm);
void set_all_enabled(bool en);
void immediate_stop();
void update_mode_b();
void Speedup(bool left, bool right);
void Stop();
float get_dist(int trig, int echo);
void updateLidarTask(unsigned long now);
void updateUltrasonicTask(unsigned long now);

// ============================================================
//  SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    Serial.println(F("Robot ready."));
    Serial.println(F("Mode A: 1 byte  (0x00 STOP | 0x01 FWD | 0x02 REV | 0x03 LEFT | 0x04 RIGHT | 0xFF HALT)"));
    Serial.println(F("Mode B: 4 bytes (0xA0 | dir | steps_hi | steps_lo)"));

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    frontRight.set_enabled(false); frontRight.set_direction(false);
    backRight.set_enabled(false);  backRight.set_direction(false);
    frontLeft.set_enabled(false);  frontLeft.set_direction(false);
    backLeft.set_enabled(false);   backLeft.set_direction(false);

    pinMode(SHT_LOX, OUTPUT);
    digitalWrite(SHT_LOX, HIGH);
    delay(10);
    if (!lox.begin()) {
        Serial.println(F("WARNING: VL53L0X not found"));
    }

    pinMode(FRONT_TRIG, OUTPUT); pinMode(FRONT_ECHO, INPUT);
    pinMode(BL_TRIG,    OUTPUT); pinMode(BL_ECHO,    INPUT);
    pinMode(BR_TRIG,    OUTPUT); pinMode(BR_ECHO,    INPUT);

    lastRampMs       = millis();
    lastLidarMs      = millis();
    lastUltrasonicMs = millis();
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
    unsigned long now = millis();

    if (modeBActive) {
        update_mode_b();
    } else {
        update_motors();
    }

    read_serial();

    if (!modeBActive) {
        if ((now - lastRampMs >= ACCEL_INTERVAL) && on == 1)       Speedup(ls, rs);
        if ((now - lastRampMs >= ACCEL_INTERVAL) && stopFlag == 1) Stop();
    }

    updateLidarTask(now);

    // Skip blocking ultrasonic reads during Mode B to avoid disrupting step timing
    if (!modeBActive) {
        updateUltrasonicTask(now);
    }
}

// ============================================================
//  MOTOR UPDATE
// ============================================================
void update_motors() {
    backRight.update();
    frontRight.update();
    backLeft.update();
    frontLeft.update();
}

// ============================================================
//  SERIAL PARSER
// ============================================================
void read_serial() {
    while (Serial.available()) {
        byte b = Serial.read();

        switch (packetState) {
            case PKT_IDLE:
                if (b == MODE_B_HEADER) {
                    packetState = PKT_BYTE1;
                } else {
                    last_command      = b;
                    last_command_time = millis();
                    handle_mode_a(b);
                }
                break;

            case PKT_BYTE1:
                if (b == MODE_B_HEADER) break;
                pktDir      = b;
                packetState = PKT_BYTE2;
                break;

            case PKT_BYTE2:
                pktHigh     = b;
                packetState = PKT_BYTE3;
                break;

            case PKT_BYTE3: {
                uint16_t steps = ((uint16_t)pktHigh << 8) | b;
                handle_mode_b(pktDir, steps);
                packetState = PKT_IDLE;
                break;
            }
        }
    }

    if ((millis() - last_command_time) > command_timeout &&
        (leftSpeed != 0 || rightSpeed != 0)) {
        stopFlag = 1;
        on       = 0;
        Stop();
    }
}

// ============================================================
//  MODE A HANDLER
// ============================================================
void handle_mode_a(byte cmd) {
    if (modeBActive) {
        modeBActive = false;
        immediate_stop();
        Serial.println(F("Mode B interrupted"));
    }

    switch (cmd) {
        case STOP:
            stopFlag = 1;
            on = 0;
            Stop();
            break;

        case FWD:
            set_all_enabled(true);
            if (leftSpeed >= 3 && rightSpeed >= 3) break;
            if (leftSpeed  < 3) leftSpeed++;
            if (rightSpeed < 3) rightSpeed++;
            on = 1; stopFlag = 0;
            ls = (leftSpeed  >= 0);
            rs = (rightSpeed <  0);
            slowSpeed = ACCEL_INCREMENT;
            Speedup(ls, rs);
            break;

        case REV:
            set_all_enabled(true);
            if (leftSpeed <= -3 && rightSpeed <= -3) break;
            if (leftSpeed  > -3) leftSpeed--;
            if (rightSpeed > -3) rightSpeed--;
            on = 1; stopFlag = 0;
            ls = (leftSpeed  >= 0);
            rs = (rightSpeed <  0);
            slowSpeed = ACCEL_INCREMENT;
            Speedup(ls, rs);
            break;

        case RIGHT:
            set_all_enabled(true);
            if (leftSpeed >= 3 && rightSpeed <= -3) break;
            if (leftSpeed  < 3) leftSpeed++;
            if (rightSpeed > -3) rightSpeed--;
            on = 1; stopFlag = 0;
            ls = (leftSpeed  >= 0);
            rs = (rightSpeed <  0);
            slowSpeed = ACCEL_INCREMENT;
            Speedup(ls, rs);
            break;

        case LEFT:
            set_all_enabled(true);
            if (rightSpeed >= 3 && leftSpeed <= -3) break;
            if (rightSpeed < 3) rightSpeed++;
            if (leftSpeed > -3) leftSpeed--;
            on = 1; stopFlag = 0;
            ls = (leftSpeed  >= 0);
            rs = (rightSpeed <  0);
            slowSpeed = ACCEL_INCREMENT;
            Speedup(ls, rs);
            break;

        default:
            digitalWrite(LED_BUILTIN, LOW);
            stopFlag = 1;
            on = 0;
            Stop();
            break;
    }
}

// ============================================================
//  MODE B HANDLER
// ============================================================
void handle_mode_b(byte dir, uint16_t steps) {
    immediate_stop();

    Serial.print(F("Mode B: dir=0x")); Serial.print(dir, HEX);
    Serial.print(F(" steps="));        Serial.println(steps);

    // Record start positions BEFORE enabling motors so the baseline is clean
    targetSteps = (long)steps;
    startPosFL  = frontLeft.get_position();
    startPosBL  = backLeft.get_position();
    startPosFR  = frontRight.get_position();
    startPosBR  = backRight.get_position();

    apply_direction_b(dir);
    set_all_speed(MODE_B_RPM);
    set_all_enabled(true);

    modeBActive = (targetSteps > 0);

    if (!modeBActive) {
        Serial.println(F("Mode B: running indefinitely (steps=0)"));
    }
}

void apply_direction_b(byte dir) {
    bool leftFwd, rightFwd;
    switch (dir) {
        case FWD:   leftFwd = true;  rightFwd = false; break;
        case REV:   leftFwd = false; rightFwd = true;  break;
        case LEFT:  leftFwd = false; rightFwd = false; break;
        case RIGHT: leftFwd = true;  rightFwd = true;  break;
        default:    leftFwd = true;  rightFwd = false; break;
    }
    frontLeft.set_direction(leftFwd);
    backLeft.set_direction(leftFwd);
    frontRight.set_direction(rightFwd);
    backRight.set_direction(rightFwd);
}

void set_all_speed(long rpm) {
    frontLeft.set_speed(rpm);
    backLeft.set_speed(rpm);
    frontRight.set_speed(rpm);
    backRight.set_speed(rpm);
}

void set_all_enabled(bool en) {
    frontLeft.set_enabled(en);
    backLeft.set_enabled(en);
    frontRight.set_enabled(en);
    backRight.set_enabled(en);
}

// ============================================================
//  MODE B: per-motor step check
// ============================================================
void update_mode_b() {
    bool allDone = true;

    if (abs(frontLeft.get_position()  - startPosFL) < targetSteps) {
        frontLeft.update();  allDone = false;
    } else {
        frontLeft.set_enabled(false);
    }

    if (abs(backLeft.get_position()   - startPosBL) < targetSteps) {
        backLeft.update();   allDone = false;
    } else {
        backLeft.set_enabled(false);
    }

    if (abs(frontRight.get_position() - startPosFR) < targetSteps) {
        frontRight.update(); allDone = false;
    } else {
        frontRight.set_enabled(false);
    }

    if (abs(backRight.get_position()  - startPosBR) < targetSteps) {
        backRight.update();  allDone = false;
    } else {
        backRight.set_enabled(false);
    }

    if (allDone) {
        modeBActive = false;
        Serial.println(F("Mode B: done"));
        Serial.print(F("  FL=")); Serial.print(abs(frontLeft.get_position()  - startPosFL));
        Serial.print(F("  BL=")); Serial.print(abs(backLeft.get_position()   - startPosBL));
        Serial.print(F("  FR=")); Serial.print(abs(frontRight.get_position() - startPosFR));
        Serial.print(F("  BR=")); Serial.println(abs(backRight.get_position() - startPosBR));
    }
}

// ============================================================
//  IMMEDIATE STOP (no ramp)
// ============================================================
void immediate_stop() {
    on = 0; stopFlag = 0;
    leftSpeed = 0; rightSpeed = 0;
    idle_left_speed = 1; idle_right_speed = 1;
    slowSpeed = ACCEL_INCREMENT;
    set_all_enabled(false);
}

// ============================================================
//  MODE A: SPEEDUP RAMP
// ============================================================
void Speedup(bool left, bool right) {
    int targetSpeedL, targetSpeedR;

    const int slowSpeedValue   = maxSpeed;
    const int mediumSpeedValue = maxSpeed * 2;
    const int fastSpeedValue   = maxSpeed * 3;

    switch (abs(leftSpeed)) {
        case 1:  targetSpeedL = slowSpeedValue;   break;
        case 2:  targetSpeedL = mediumSpeedValue; break;
        case 3:  targetSpeedL = fastSpeedValue;   break;
        default: targetSpeedL = 0;                break;
    }
    switch (abs(rightSpeed)) {
        case 1:  targetSpeedR = slowSpeedValue;   break;
        case 2:  targetSpeedR = mediumSpeedValue; break;
        case 3:  targetSpeedR = fastSpeedValue;   break;
        default: targetSpeedR = 0;                break;
    }

    if (leftSpeed  < 0) targetSpeedL *= -1;
    if (rightSpeed < 0) targetSpeedR *= -1;

    if (idle_left_speed < targetSpeedL) {
        idle_left_speed += slowSpeed;
        if (idle_left_speed > targetSpeedL) idle_left_speed = targetSpeedL;
    } else if (idle_left_speed > targetSpeedL) {
        idle_left_speed -= slowSpeed;
        if (idle_left_speed < targetSpeedL) idle_left_speed = targetSpeedL;
    }

    if (idle_right_speed < targetSpeedR) {
        idle_right_speed += slowSpeed;
        if (idle_right_speed > targetSpeedR) idle_right_speed = targetSpeedR;
    } else if (idle_right_speed > targetSpeedR) {
        idle_right_speed -= slowSpeed;
        if (idle_right_speed < targetSpeedR) idle_right_speed = targetSpeedR;
    }

    bool leftReached  = (idle_left_speed  == targetSpeedL);
    bool rightReached = (idle_right_speed == targetSpeedR);

    lastRampMs = millis();

    if ((idle_left_speed >= 0) == left) {
        frontLeft.set_direction(left);
        backLeft.set_direction(left);
    } else {
        frontLeft.set_direction(!left);
        backLeft.set_direction(!left);
    }

    if ((idle_right_speed <= 0) == right) {
        backRight.set_direction(right);
        frontRight.set_direction(right);
    } else {
        backRight.set_direction(!right);
        frontRight.set_direction(!right);
    }

    if (leftReached && rightReached) {
        frontLeft.set_speed(abs(targetSpeedL));
        backLeft.set_speed(abs(targetSpeedL));
        backRight.set_speed(abs(targetSpeedR));
        frontRight.set_speed(abs(targetSpeedR));

#if PRINT_SENSOR_DEBUG
        Serial.print(F("max reached targetL: ")); Serial.print(targetSpeedL);
        Serial.print(F("   targetR: "));          Serial.println(targetSpeedR);
#endif
        on = 0;
        slowSpeed = ACCEL_INCREMENT;
        return;
    }

    frontLeft.set_speed(abs(idle_left_speed));
    backLeft.set_speed(abs(idle_left_speed));
    backRight.set_speed(abs(idle_right_speed));
    frontRight.set_speed(abs(idle_right_speed));

#if PRINT_SENSOR_DEBUG
    Serial.print(F("idleL: ")); Serial.print(idle_left_speed);
    Serial.print(F("  --> ")); Serial.println(targetSpeedL);
    Serial.print(F("idleR: ")); Serial.print(idle_right_speed);
    Serial.print(F("  --> ")); Serial.println(targetSpeedR);
    Serial.print(F("slow speed: ")); Serial.println(slowSpeed);
    Serial.println(F("----------------"));
#endif

    slowSpeed = min(slowSpeed * 2, ACCEL_MAX);
}

// ============================================================
//  MODE A: STOP RAMP
// ============================================================
void Stop() {
    if (abs(idle_left_speed) <= ACCEL_INCREMENT && abs(idle_right_speed) <= ACCEL_INCREMENT) {
        idle_left_speed  = 1;
        idle_right_speed = 1;
        slowSpeed  = ACCEL_INCREMENT;
        stopFlag   = 0;
        leftSpeed  = 0;
        rightSpeed = 0;

        backRight.set_speed(0);
        frontRight.set_speed(0);
        backLeft.set_speed(0);
        frontLeft.set_speed(0);

#if PRINT_SENSOR_DEBUG
        Serial.println(F("--STOPPED--"));
#endif
        return;
    }

    lastRampMs = millis();

#if PRINT_SENSOR_DEBUG
    Serial.println(F("Cutting speed in half..."));
#endif

    idle_left_speed  /= 2;
    idle_right_speed /= 2;

    frontLeft.set_speed(abs(idle_left_speed));
    backLeft.set_speed(abs(idle_left_speed));
    backRight.set_speed(abs(idle_right_speed));
    frontRight.set_speed(abs(idle_right_speed));
}

// ============================================================
//  ULTRASONIC
// ============================================================
float get_dist(int trig, int echo) {
    digitalWrite(trig, LOW);
    delayMicroseconds(2);
    digitalWrite(trig, HIGH);
    delayMicroseconds(10);
    digitalWrite(trig, LOW);

    long duration = pulseIn(echo, HIGH, ULTRASONIC_TIMEOUT_US);
    return (duration == 0) ? -1.0f : duration * 0.034f / 2.0f;
}

// ============================================================
//  LIDAR TASK
// ============================================================
void updateLidarTask(unsigned long now) {
    if (now - lastLidarMs < LIDAR_INTERVAL_MS) return;
    lastLidarMs = now;

    VL53L0X_RangingMeasurementData_t measure;
    lox.rangingTest(&measure, false);

    if (measure.RangeStatus != 4) {
        lidarDistanceMm = measure.RangeMilliMeter;
        lidarValid      = true;
    } else {
        lidarValid = false;
    }

#if PRINT_SENSOR_DEBUG
    Serial.print(F("LiDAR: "));
    if (lidarValid) { Serial.print(lidarDistanceMm); Serial.println(F(" mm")); }
    else            { Serial.println(F("out of range")); }
#endif
}

// ============================================================
//  ULTRASONIC TASK
//  Not called during Mode B — pulseIn() would disrupt step timing.
// ============================================================
void updateUltrasonicTask(unsigned long now) {
    if (now - lastUltrasonicMs < ULTRASONIC_STEP_INTERVAL_MS) return;
    lastUltrasonicMs = now;

    switch (ultrasonicPhase) {
        case 0: frontDistCm = get_dist(FRONT_TRIG, FRONT_ECHO); ultrasonicPhase = 1; break;
        case 1: blDistCm    = get_dist(BL_TRIG,    BL_ECHO);    ultrasonicPhase = 2; break;
        case 2:
            brDistCm        = get_dist(BR_TRIG,    BR_ECHO);
            ultrasonicPhase = 0;

            // ── sensor report for Raspberry Pi ──────────────────────────
            {
                float lidarCm = lidarValid ? lidarDistanceMm / 10.0f : -1.0f;
                bool obstacle = (frontDistCm > 0 && frontDistCm < OBSTACLE_THRESHOLD_CM)
                             || (lidarValid   && lidarCm       < OBSTACLE_THRESHOLD_CM);

                Serial.print(F("Ultrasonic: "));
                Serial.print(frontDistCm, 1);
                Serial.print(F(" cm | LiDAR: "));
                if (lidarValid) Serial.print(lidarCm, 1);
                else            Serial.print(F("-1"));
                Serial.print(F(" cm | "));
                Serial.println(obstacle ? F("OBSTACLE") : F("NORMAL"));
            }
            // ─────────────────────────────────────────────────────────────

#if PRINT_SENSOR_DEBUG
            Serial.print(frontDistCm); Serial.print(F(", "));
            Serial.print(blDistCm);    Serial.print(F(", "));
            Serial.println(brDistCm);
#endif
            break;
    }
}
