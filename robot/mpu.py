import smbus2 as smbus
import math
import time

class MPU6050:
    def __init__(self, bus=1, address=0x68):
        self.bus = smbus.SMBus(bus)
        self.address = address
        self.initialize_mpu()
        
    def initialize_mpu(self):
        self.bus.write_byte_data(self.address, 0x6B, 0x00)  # Wake up
        self.bus.write_byte_data(self.address, 0x1B, 0x08)  # Gyro ±500°/s
        self.bus.write_byte_data(self.address, 0x1C, 0x08)  # Accel ±4g
        self.calibrate_gyro()
        
    def calibrate_gyro(self, samples=1000):
        print("Calibrating gyroscope...")
        offsets = [0, 0, 0]
        for _ in range(samples):
            for i in range(3):
                offsets[i] += self.read_raw_data(0x43 + i*2)
            time.sleep(0.002)
        self.gyro_offsets = [x/samples for x in offsets]
        
    def read_raw_data(self, addr):
        high = self.bus.read_byte_data(self.address, addr)
        low = self.bus.read_byte_data(self.address, addr+1)
        value = (high << 8) + low
        return value - 65536 if value > 32768 else value
    
    def get_accel_data(self):
        scale = 8192.0  # ±4g range
        return [self.read_raw_data(addr)/scale for addr in [0x3B, 0x3D, 0x3F]]
    
    def get_gyro_data(self):
        scale = 65.5  # ±500°/s range
        return [(self.read_raw_data(addr) - offset)/scale 
               for addr, offset in zip([0x43, 0x45, 0x47], self.gyro_offsets)]
    
    def get_angles(self):
        accel_x, accel_y, accel_z = self.get_accel_data()
        gyro_x, gyro_y, _ = self.get_gyro_data()
        
        # Calculate angles from accelerometer
        accel_angle_x = math.atan2(accel_y, accel_z) * (180/math.pi)
        accel_angle_y = math.atan2(-accel_x, math.hypot(accel_y, accel_z)) * (180/math.pi)
        
        # Complementary filter
        dt = 0.01  # Fixed time step
        alpha = 0.96
        self.angle_x = alpha*(self.angle_x + gyro_x*dt) + (1-alpha)*accel_angle_x
        self.angle_y = alpha*(self.angle_y + gyro_y*dt) + (1-alpha)*accel_angle_y
        
        return self.angle_x, self.angle_y