from mpu6050 import MPU6050
from pid import PID
from motor import Motor
import time
import math

# Complementary filter constants
ALPHA = 0.98
DT = 0.02  # 50Hz loop

# Motor pins (adjust according to your wiring)
MOTOR1_PINS = (22, 23, 12)  # IN1, IN2, EN
MOTOR2_PINS = (24, 25, 13)  # IN1, IN2, EN

# PID constants (you'll need to tune these)
KP = 15.0
KI = 0.0
KD = 0.1

def calculate_angle(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, dt, prev_angle):
    # Calculate angle from accelerometer
    acc_angle_x = math.atan2(acc_y, math.sqrt(acc_z**2 + acc_x**2)) * (180/math.pi)
    
    # Calculate angle from gyroscope
    gyro_rate = gyro_x
    gyro_angle = prev_angle + gyro_rate * dt
    
    # Complementary filter
    angle = ALPHA * gyro_angle + (1 - ALPHA) * acc_angle_x
    
    return angle

def main():
    try:
        print("Initializing components...")
        
        # Initialize IMU
        imu = MPU6050()
        print("IMU initialized successfully")
        
        # Initialize PID
        pid = PID(KP, KI, KD)
        
        # Initialize Motors
        print("Initializing motors...")
        motor1 = Motor(*MOTOR1_PINS)
        motor2 = Motor(*MOTOR2_PINS)
        atexit.register(Motor.cleanup)  # Ensure cleanup on exit
        
        print("Balancing robot starting...")
        time.sleep(1)  # Wait for IMU to stabilize
        
        target_angle = 0  # The angle we want to maintain (upright)
        current_angle = 0
        
        while True:
            start_time = time.time()
            
            # Read IMU data
            acc_x, acc_y, acc_z = imu.get_accel_data()
            gyro_x, gyro_y, gyro_z = imu.get_gyro_data()
            
            # Calculate current angle
            current_angle = calculate_angle(
                acc_x, acc_y, acc_z,
                gyro_x, gyro_y, gyro_z,
                DT, current_angle
            )
            
            # Compute PID output
            error = target_angle - current_angle
            output = pid.compute(error)
            
            # Constrain output and apply to motors
            output = max(-100, min(100, output))
            
            if abs(output) > 5:  # Deadband to prevent jitter
                if output > 0:
                    motor1.move_forward(output)
                    motor2.move_forward(output)
                else:
                    motor1.move_backward(-output)
                    motor2.move_backward(-output)
            else:
                motor1.stop()
                motor2.stop()
            
            # Maintain loop timing
            elapsed = time.time() - start_time
            if elapsed < DT:
                time.sleep(DT - elapsed)
                
    except KeyboardInterrupt:
        print("\nStopping robot...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        Motor.cleanup()

if __name__ == "__main__":
    main()