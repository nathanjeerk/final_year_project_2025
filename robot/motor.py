import RPi.GPIO as GPIO

class Motor:
    def __init__(self, in1, in2, pwm, encoder_a, encoder_b):
        self.in1 = in1
        self.in2 = in2
        self.pwm_pin = pwm
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup([in1, in2, pwm], GPIO.OUT)
        GPIO.setup([encoder_a, encoder_b], GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        self.pwm = GPIO.PWM(pwm, 1000)  # 1kHz frequency
        self.pwm.start(0)
        
        # Encoder setup
        self.encoder_pos = 0
        GPIO.add_event_detect(encoder_a, GPIO.BOTH, callback=self._update_encoder)
    
    def _update_encoder(self, channel):
        a_state = GPIO.input(self.encoder_a)
        b_state = GPIO.input(self.encoder_b)
        self.encoder_pos += 1 if a_state == b_state else -1
    
    def set_speed(self, speed):
        if speed > 0:
            GPIO.output(self.in1, GPIO.HIGH)
            GPIO.output(self.in2, GPIO.LOW)
        elif speed < 0:
            GPIO.output(self.in1, GPIO.LOW)
            GPIO.output(self.in2, GPIO.HIGH)
        else:
            GPIO.output(self.in1, GPIO.LOW)
            GPIO.output(self.in2, GPIO.LOW)
            
        self.pwm.ChangeDutyCycle(abs(speed))
    
    def cleanup(self):
        self.pwm.stop()
        GPIO.cleanup()