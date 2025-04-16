import time

class PID:
    def __init__(self, kp, ki, kd, setpoint=0, output_limits=(-100, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.limits = output_limits
        
        self._last_error = 0
        self._integral = 0
        self._last_time = time.time()
    
    def compute(self, input_value):
        now = time.time()
        dt = now - self._last_time
        if dt <= 0:
            return 0
            
        error = self.setpoint - input_value
        self._integral += error * dt
        derivative = (error - self._last_error) / dt
        
        # PID calculation
        output = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)
        
        # Apply output limits
        output = max(self.limits[0], min(self.limits[1], output))
        
        self._last_error = error
        self._last_time = now
        
        return output
    
    def reset(self):
        self._integral = 0
        self._last_error = 0