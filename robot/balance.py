import time
import math
import threading
import smbus2
import RPi.GPIO as GPIO
import os
import signal
import sys

# === MPU6050 Constants ===
MPU_ADDR = 0x68
bus = smbus2.SMBus(1)
OFFSET_FILE = "balance_offset.txt"

# === Motor GPIO Pins ===
IN1, IN2 = 17, 18
IN3, IN4 = 22, 23
ENA, ENB = 12, 13

# === PID Parameters
Kp, Ki, Kd = 30.0, 0.0, 1.5
alpha = 0.98

# === Shared state
angle_display = 0.0
pid_output_display = 0.0
current_offset = 0.0
pid_lock = threading.Lock()

# === GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(0)
pwmB.start(0)

# === Cleanup on exit
def stop_motors():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)
    GPIO.cleanup()

def handle_exit(sig, frame):
    print("Stopping motors safely...")
    stop_motors()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# === MPU Functions
def mpu_init():
    bus.write_byte_data(MPU_ADDR, 0x6B, 0)

def read_mpu():
    data = bus.read_i2c_block_data(MPU_ADDR, 0x3B, 14)
    ax = (data[0] << 8) | data[1]
    ay = (data[2] << 8) | data[3]
    az = (data[4] << 8) | data[5]
    gx = (data[8] << 8) | data[9]
    if ax > 32767: ax -= 65536
    if ay > 32767: ay -= 65536
    if az > 32767: az -= 65536
    if gx > 32767: gx -= 65536
    ax /= 16384.0
    ay /= 16384.0
    az /= 16384.0
    gx /= 131.0
    return ax, ay, az, gx

def compute_accel_angle(ax, az):
    return math.atan2(ax, az) * 180 / math.pi

# === Calibration
def calibrate_offset():
    print("[CALIBRATION] Keep robot upright...")
    time.sleep(2)
    samples = []
    for _ in range(100):
        ax, _, az, _ = read_mpu()
        angle = compute_accel_angle(ax, az)
        samples.append(angle)
        time.sleep(0.01)
    offset = sum(samples) / len(samples)
    print(f"[CALIBRATION COMPLETE] Offset: {offset:.2f}°")
    with open(OFFSET_FILE, "w") as f:
        f.write(str(offset))
    return offset

def load_offset():
    if os.path.exists(OFFSET_FILE):
        return float(open(OFFSET_FILE).read())
    return calibrate_offset()

def recalibrate_offset():
    global current_offset
    current_offset = calibrate_offset()
    return current_offset

def get_offset():
    return round(current_offset, 2)

# === PID
def pid(setpoint, measured, prev_err, integral, dt):
    with pid_lock:
        kp, ki, kd = Kp, Ki, Kd
    error = setpoint - measured
    integral += error * dt
    derivative = (error - prev_err) / dt
    output = kp * error + ki * integral + kd * derivative
    return output, error, integral

def set_pid(kp, ki, kd):
    global Kp, Ki, Kd
    with pid_lock:
        Kp, Ki, Kd = kp, ki, kd

def get_pid():
    with pid_lock:
        return Kp, Ki, Kd

# === Motor Driver
def set_motors(output, mode="BALANCE"):
    speed = max(min(abs(output), 100), 0)
    if mode == "LEFT":
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
    elif mode == "RIGHT":
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    elif mode == "FORWARD":
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
    elif mode == "BACKWARD":
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    elif mode == "BALANCE":
        if output > 0:
            GPIO.output(IN1, GPIO.HIGH)
            GPIO.output(IN2, GPIO.LOW)
            GPIO.output(IN3, GPIO.HIGH)
            GPIO.output(IN4, GPIO.LOW)
        elif output < 0:
            GPIO.output(IN1, GPIO.LOW)
            GPIO.output(IN2, GPIO.HIGH)
            GPIO.output(IN3, GPIO.LOW)
            GPIO.output(IN4, GPIO.HIGH)
        else:
            GPIO.output(IN1, GPIO.LOW)
            GPIO.output(IN2, GPIO.LOW)
            GPIO.output(IN3, GPIO.LOW)
            GPIO.output(IN4, GPIO.LOW)

    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)

# === Balancing Loop
def balance_loop(get_drive_mode):
    global angle_display, pid_output_display, current_offset
    mpu_init()
    current_offset = load_offset()
    setpoint = 0
    prev_angle = prev_error = integral = 0
    last_time = time.time()

    while True:
        ax, _, az, gx = read_mpu()
        now = time.time()
        dt = now - last_time
        last_time = now

        accel_angle = compute_accel_angle(ax, az)
        gyro_angle = prev_angle + gx * dt
        angle = alpha * gyro_angle + (1 - alpha) * accel_angle
        corrected = angle - current_offset

        output, prev_error, integral = pid(setpoint, corrected, prev_error, integral, dt)

        # Live monitoring
        angle_display = corrected
        pid_output_display = output

        mode = get_drive_mode()
        if mode == "LEFT":
            set_motors(output, "LEFT")
        elif mode == "RIGHT":
            set_motors(output, "RIGHT")
        elif mode == "FORWARD":
            set_motors(output, "FORWARD")
        elif mode == "BACKWARD":
            set_motors(output, "BACKWARD")
        else:
            set_motors(output, "BALANCE")

        prev_angle = angle
        time.sleep(0.01)

# === Start balance thread
def start_balance_loop(get_drive_mode):
    t = threading.Thread(target=balance_loop, args=(get_drive_mode,), daemon=True)
    t.start()

def get_live_data():
    return {
        "angle": round(angle_display, 2),
        "pid_output": round(pid_output_display, 2),
        "kp": Kp,
        "ki": Ki,
        "kd": Kd
    }
