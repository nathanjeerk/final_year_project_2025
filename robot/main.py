from balance import start_balance_loop
from detect import start_pose_detection
from server import run_flask, set_drive_mode_callback

# Shared variable to hold current movement mode from joystick
drive_mode = "STOP"

# Function to provide current mode to balance.py
def get_drive_mode():
    return drive_mode

# Function for server.py to update drive_mode
def set_drive_mode(new_mode):
    global drive_mode
    drive_mode = new_mode

if __name__ == "__main__":
    # Set callback to let server control movement
    set_drive_mode_callback(set_drive_mode)

    # Start modules
    start_pose_detection()
    start_balance_loop(get_drive_mode)
    run_flask()
