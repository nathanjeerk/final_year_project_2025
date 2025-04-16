from mpu import MPU6050
from pid import PID
from motor import Motor
import time

class BalancingRobot:
    def __init__(self):
        # Initialize components
        self.imu = MPU6050()
        self.pid = PID(kp=15, ki=0.5, kd=0.1, setpoint=0)  # Tune these values!
        
        # Motor pins (adjust to your wiring)
        self.left_motor = Motor(in1=17, in2=27, pwm=22, encoder_a=23, encoder_b=24)
        self.right_motor = Motor(in1=5, in2=6, pwm=13, encoder_a=19, encoder_b=26)
        
        # Balance parameters
        self.target_angle = 0  # Degrees (upright position)
        self.running = True
    
    def balance_loop(self):
        try:
            while self.running:
                # Get current angle
                angle_x, _ = self.imu.get_angles()  # We only need pitch (x-axis)
                
                # PID calculation
                output = self.pid.compute(angle_x)
                
                # Apply to motors
                self.left_motor.set_speed(output)
                self.right_motor.set_speed(output)
                
                time.sleep(0.01)  # 100Hz loop
                
        except KeyboardInterrupt:
            self.running = False
        finally:
            self.left_motor.cleanup()
            self.right_motor.cleanup()

if __name__ == "__main__":
    robot = BalancingRobot()
    print("Starting balance loop...")
    robot.balance_loop()
    print("Robot stopped")