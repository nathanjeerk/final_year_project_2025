from flask import Flask, render_template, request, Response, jsonify
from detect import get_latest_frame, get_latest_label
from balance import get_live_data, set_pid, get_pid, recalibrate_offset, get_offset

app = Flask(__name__)

# Shared state for movement control
drive_mode = "STOP"
drive_mode_callback = lambda: "STOP"

def get_drive_mode():
    return drive_mode

def set_drive_mode_callback(callback):
    global drive_mode_callback
    drive_mode_callback = callback

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/control")
def control():
    cmd = request.args.get("dir", "").upper()
    if cmd in ["FORWARD", "LEFT", "RIGHT", "BACKWARD", "STOP"]:
        drive_mode_callback(cmd)
        return f"Direction set to {cmd}"
    return "Invalid direction", 400

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = get_latest_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    import cv2
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/pose_label")
def pose_label():
    return jsonify({"label": get_latest_label()})

@app.route("/data")
def data():
    return jsonify(get_live_data())

@app.route("/get_pid")
def get_pid_values():
    kp, ki, kd = get_pid()
    return jsonify({"kp": kp, "ki": ki, "kd": kd})

@app.route("/set_pid", methods=["POST"])
def update_pid():
    try:
        kp = float(request.form["kp"])
        ki = float(request.form["ki"])
        kd = float(request.form["kd"])
        set_pid(kp, ki, kd)
        return "PID updated"
    except:
        return "Invalid PID values", 400

@app.route("/get_offset")
def get_offset_value():
    return jsonify({"offset": round(get_offset(), 2)})

@app.route("/calibrate", methods=["POST"])
def do_calibration():
    offset = recalibrate_offset()
    return jsonify({"message": f"Calibrated: {round(offset, 2)}°"})


def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False)
