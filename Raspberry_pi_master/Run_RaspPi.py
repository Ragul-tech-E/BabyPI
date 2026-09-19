#!/usr/bin/env python3
# ============================================================
#                  BABY MONITOR SYSTEM
# ============================================================
#
# Raspberry Pi Zero 2 W / Raspberry Pi OS (latest)
#
# FUNCTIONS
# 1. INMP441 automatic sound detection
# 2. Automatic baby-cry recording
# 3. Software audio gain = 6x
# 4. 4 second recording
# 5. TFLite baby audio classification
# 6. OLED status / prediction
# 7. Prediction shown for 8 seconds
# 8. Asphyxia -> esp32_f.py
# 9. GPIO14 single press -> complete program restart
# 10. GPIO14 double press -> battery display
# 11. GPIO14 long press -> 1,2,3 shutdown countdown
#     EACH NUMBER = 2 seconds, TOTAL = 6 seconds
# 12. ADS1115 A1 battery monitoring
# 13. Low battery <= 3.55V -> LOW BATTERY for 5 sec
#     then 1,2,3 shutdown countdown
# 14. OLED is cleared/hidden before Linux poweroff
#
# IMPORTANT:
# Run this program from the venv:
# /home/babypi/baby_model/.venv/bin/python /home/babypi/baby_model/run_f.py
#
# ============================================================

import os
import sys
import json
import time
import subprocess
import threading
import collections

import numpy as np
import librosa
from ai_edge_litert import interpreter as tflite

import RPi.GPIO as GPIO
import board
import busio

from adafruit_ads1x15 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
from adafruit_ads1x15 import ads1x15

from luma.core.interface.serial import i2c as oled_i2c
from luma.oled.device import sh1106

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = "/home/babypi/baby_model"

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_TFLITE = os.path.join(MODEL_DIR, "baby_audio_strong.tflite")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
NORMALIZATION_PATH = os.path.join(MODEL_DIR, "normalization.json")

ESP32_SCRIPT = os.path.join(BASE_DIR, "esp32_f.py")

# Current interpreter is deliberately used.
# When systemd starts this file with .venv/bin/python,
# sys.executable is the venv Python.
PYTHON_EXECUTABLE = sys.executable


# ============================================================
# AUDIO
# ============================================================

AUDIO_DEVICE = "plughw:0,0"
AUDIO_FORMAT = "S32_LE"
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000

AUDIO_GAIN = 6.0

RECORD_SECONDS = 4.0

SOUND_RMS_THRESHOLD = 0.035
SOUND_PEAK_THRESHOLD = 0.10

PRE_TRIGGER_SECONDS = 0.30
DETECTION_COOLDOWN = 0.5


# ============================================================
# PREDICTION
# ============================================================

PREDICTION_DISPLAY_SECONDS = 8.0


# ============================================================
# BATTERY
# ============================================================

BATTERY_LOW_VOLTAGE = 3.55
BATTERY_EMPTY_VOLTAGE = 3.45
BATTERY_FULL_VOLTAGE = 4.07
BATTERY_CHECK_INTERVAL = 2.0

LOW_BATTERY_DISPLAY_SECONDS = 5.0

# Battery divider:
# Battery -> 22K -> ADS A1 -> 10K -> GND
R_TOP = 22000.0
R_BOTTOM = 10000.0

DIVIDER_RATIO = (R_TOP + R_BOTTOM) / R_BOTTOM

CALIBRATION_FACTOR = 0.98541


# ============================================================
# BUTTON
# ============================================================

BUTTON_PIN = 14

DEBOUNCE_TIME = 0.05
MULTI_PRESS_TIME = 0.40
LONG_PRESS_TIME = 1.5


# ============================================================
# SHUTDOWN
# ============================================================

# User requested 1 -> 2 -> 3.
# Each number remains on OLED for 2 seconds.
SHUTDOWN_STEP_SECONDS = 2.0
SHUTDOWN_COUNTDOWN_SECONDS = 3


# ============================================================
# OLED
# ============================================================

OLED_WIDTH = 128
OLED_HEIGHT = 64
OLED_ADDRESS = 0x3C

FONT_PATH = (
    "/usr/share/fonts/truetype/dejavu/"
    "DejaVuSans-Bold.ttf"
)


# ============================================================
# GLOBAL STATE
# ============================================================

running = True
asphyxia_mode = False
esp32_process = None

low_battery_shutdown = False
poweroff_requested = False
reset_in_progress = False
processing_prediction = False
shutdown_in_progress = False

shutdown_lock = threading.Lock()

last_prediction = "READY"
last_confidence = 0.0

last_battery_voltage = 0.0
last_battery_percent = 0

oled = None
OLED_FONT_BIG = None
OLED_FONT_MEDIUM = None
OLED_FONT_SMALL = None

i2c = None
ads = None
battery_channel = None

audio_process = None

# Prevent simultaneous model inference.
prediction_lock = threading.Lock()


# ============================================================
# STOP BOOT OLED ANIMATION
# ============================================================
#
# The boot animation is normally started by:
#   babypi-oled.service
#
# This monitor service stops it after the Linux boot screen
# has finished. The stop operation is intentionally safe if
# the boot service is already finished.
# ============================================================

def stop_boot_oled_service():
    try:
        result = subprocess.run(
            [
                "systemctl",
                "stop",
                "babypi-oled.service"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3
        )

        if result.returncode == 0:
            print("[BabyPi] Boot OLED animation stopped.")
        else:
            print(
                "[BabyPi] Boot OLED service was not running "
                "or could not be stopped."
            )

    except Exception as e:
        print("[BabyPi] Boot OLED stop error:", e)


# ============================================================
# INITIALIZE OLED
# ============================================================

print()
print("=" * 70)
print("INITIALIZING OLED")
print("=" * 70)

try:
    oled_serial = oled_i2c(
        port=1,
        address=OLED_ADDRESS
    )

    oled = sh1106(
        oled_serial,
        width=OLED_WIDTH,
        height=OLED_HEIGHT,
        rotate=0
    )

    oled.contrast(255)

    OLED_FONT_BIG = ImageFont.truetype(
        FONT_PATH,
        14
    )

    OLED_FONT_MEDIUM = ImageFont.truetype(
        FONT_PATH,
        11
    )

    OLED_FONT_SMALL = ImageFont.truetype(
        FONT_PATH,
        9
    )

    oled.clear()

    print("OLED OK")
    print("SH1106 128x64")
    print("Address:", hex(OLED_ADDRESS))

except Exception as e:
    print("OLED ERROR:", e)
    oled = None


# ============================================================
# OLED OFF
# ============================================================

def oled_off():
    """Clear and hide the OLED before Linux poweroff."""
    global oled

    if oled is None:
        return

    try:
        oled.clear()

        try:
            oled.hide()
        except Exception:
            pass

        print("OLED OFF")

    except Exception as e:
        print("OLED OFF ERROR:", e)


# ============================================================
# OLED BASIC TEXT
# ============================================================

def oled_text(lines, font=None):
    if oled is None:
        return

    try:
        image = Image.new(
            "1",
            (OLED_WIDTH, OLED_HEIGHT),
            0
        )

        draw = ImageDraw.Draw(image)

        if font is None:
            font = OLED_FONT_SMALL

        y = 0

        for line in lines:
            draw.text(
                (2, y),
                str(line),
                font=font,
                fill=1
            )
            y += 13

        oled.display(image)

    except Exception as e:
        print("OLED error:", e)


# ============================================================
# WAITING SCREEN
# ============================================================

def show_waiting():
    if oled is None:
        return

    try:
        image = Image.new(
            "1",
            (OLED_WIDTH, OLED_HEIGHT),
            0
        )

        draw = ImageDraw.Draw(image)

        title = "BABY MONITOR"

        box = draw.textbbox(
            (0, 0),
            title,
            font=OLED_FONT_MEDIUM
        )

        width = box[2] - box[0]

        draw.text(
            ((128 - width) // 2, 0),
            title,
            font=OLED_FONT_MEDIUM,
            fill=1
        )

        draw.line(
            (0, 14, 127, 14),
            fill=1
        )

        text = "WAITING FOR CRY"

        box = draw.textbbox(
            (0, 0),
            text,
            font=OLED_FONT_SMALL
        )

        width = box[2] - box[0]

        draw.text(
            ((128 - width) // 2, 25),
            text,
            font=OLED_FONT_SMALL,
            fill=1
        )

        draw.text(
            (19, 45),
            "LISTENING...",
            font=OLED_FONT_SMALL,
            fill=1
        )

        oled.display(image)

    except Exception as e:
        print("OLED waiting error:", e)


# ============================================================
# RECORDING SCREEN
# ============================================================

def show_recording():
    oled_text(
        [
            "BABY MONITOR",
            "",
            "CRY DETECTED",
            "",
            "RECORDING...",
            "",
            "4 SECONDS"
        ],
        OLED_FONT_MEDIUM
    )


# ============================================================
# PROCESSING SCREEN
# ============================================================

def show_processing():
    oled_text(
        [
            "BABY MONITOR",
            "",
            "PROCESSING...",
            "",
            "AI ANALYSIS",
            "",
            "PLEASE WAIT"
        ],
        OLED_FONT_MEDIUM
    )


# ============================================================
# PREDICTION SCREEN
# ============================================================

def show_prediction(prediction, confidence):
    if oled is None:
        return

    try:
        image = Image.new(
            "1",
            (128, 64),
            0
        )

        draw = ImageDraw.Draw(image)

        title = "PREDICTED CONDITION"

        box = draw.textbbox(
            (0, 0),
            title,
            font=OLED_FONT_SMALL
        )

        title_width = box[2] - box[0]

        draw.text(
            (
                max(0, (128 - title_width) // 2),
                0
            ),
            title,
            font=OLED_FONT_SMALL,
            fill=1
        )

        draw.line(
            (0, 12, 127, 12),
            fill=1
        )

        condition = str(prediction)

        if len(condition) > 18:
            condition = condition[:18]

        box = draw.textbbox(
            (0, 0),
            condition,
            font=OLED_FONT_MEDIUM
        )

        condition_width = box[2] - box[0]

        draw.text(
            (
                max(0, (128 - condition_width) // 2),
                18
            ),
            condition,
            font=OLED_FONT_MEDIUM,
            fill=1
        )

        confidence_text = f"{confidence * 100:.1f}%"

        box = draw.textbbox(
            (0, 0),
            confidence_text,
            font=OLED_FONT_SMALL
        )

        confidence_width = box[2] - box[0]

        draw.text(
            (
                (128 - confidence_width) // 2,
                37
            ),
            confidence_text,
            font=OLED_FONT_SMALL,
            fill=1
        )

        if condition.strip().lower() == "asphyxia":
            status = "EMERGENCY"
        else:
            status = "MONITORING"

        box = draw.textbbox(
            (0, 0),
            status,
            font=OLED_FONT_SMALL
        )

        status_width = box[2] - box[0]

        draw.text(
            (
                (128 - status_width) // 2,
                51
            ),
            status,
            font=OLED_FONT_SMALL,
            fill=1
        )

        oled.display(image)

    except Exception as e:
        print("OLED prediction error:", e)


# ============================================================
# ASPHYXIA SCREEN
# ============================================================

def show_asphyxia():
    if oled is None:
        return

    try:
        image = Image.new(
            "1",
            (128, 64),
            0
        )

        draw = ImageDraw.Draw(image)

        title = "ASPHYXIA"

        box = draw.textbbox(
            (0, 0),
            title,
            font=OLED_FONT_BIG
        )

        width = box[2] - box[0]

        draw.text(
            ((128 - width) // 2, 0),
            title,
            font=OLED_FONT_BIG,
            fill=1
        )

        draw.line(
            (0, 17, 127, 17),
            fill=1
        )

        draw.text(
            (17, 25),
            "EMERGENCY",
            font=OLED_FONT_MEDIUM,
            fill=1
        )

        draw.text(
            (16, 45),
            "SYSTEM ACTIVE",
            font=OLED_FONT_SMALL,
            fill=1
        )

        oled.display(image)

    except Exception as e:
        print("OLED asphyxia error:", e)


# ============================================================
# BATTERY SCREEN
# ============================================================

def show_battery(voltage, percentage):
    if oled is None:
        return

    try:
        image = Image.new(
            "1",
            (128, 64),
            0
        )

        draw = ImageDraw.Draw(image)

        title = "BATTERY"

        box = draw.textbbox(
            (0, 0),
            title,
            font=OLED_FONT_MEDIUM
        )

        width = box[2] - box[0]

        draw.text(
            ((128 - width) // 2, 0),
            title,
            font=OLED_FONT_MEDIUM,
            fill=1
        )

        voltage_text = f"{voltage:.2f} V"

        box = draw.textbbox(
            (0, 0),
            voltage_text,
            font=OLED_FONT_MEDIUM
        )

        width = box[2] - box[0]

        draw.text(
            ((128 - width) // 2, 18),
            voltage_text,
            font=OLED_FONT_MEDIUM,
            fill=1
        )

        percentage_text = f"{percentage}%"

        box = draw.textbbox(
            (0, 0),
            percentage_text,
            font=OLED_FONT_BIG
        )

        width = box[2] - box[0]

        draw.text(
            ((128 - width) // 2, 36),
            percentage_text,
            font=OLED_FONT_BIG,
            fill=1
        )

        oled.display(image)

    except Exception as e:
        print("OLED battery error:", e)


# ============================================================
# LOW BATTERY SCREEN
# ============================================================

def show_low_battery(voltage):
    oled_text(
        [
            "LOW BATTERY",
            "",
            f"{voltage:.2f} V",
            "",
            "SHUTTING DOWN",
            "",
            "PLEASE WAIT"
        ],
        OLED_FONT_MEDIUM
    )


# ============================================================
# POWER OFF SCREEN
# ============================================================

def show_powering_off(number):
    """
    Display shutdown countdown exactly as:
       1
       2
       3
    Each number remains for 2 seconds.
    """

    if oled is None:
        return

    try:
        image = Image.new(
            "1",
            (OLED_WIDTH, OLED_HEIGHT),
            0
        )

        draw = ImageDraw.Draw(image)

        title = "POWERING OFF"

        box = draw.textbbox(
            (0, 0),
            title,
            font=OLED_FONT_BIG
        )

        width = box[2] - box[0]

        draw.text(
            (
                (OLED_WIDTH - width) // 2,
                0
            ),
            title,
            font=OLED_FONT_BIG,
            fill=1
        )

        draw.line(
            (0, 18, 127, 18),
            fill=1
        )

        if number > 0:
            big_font = ImageFont.truetype(
                FONT_PATH,
                28
            )

            text = str(number)

            box = draw.textbbox(
                (0, 0),
                text,
                font=big_font
            )

            width = box[2] - box[0]

            draw.text(
                (
                    (OLED_WIDTH - width) // 2,
                    23
                ),
                text,
                font=big_font,
                fill=1
            )

        else:
            text = "POWER OFF"

            box = draw.textbbox(
                (0, 0),
                text,
                font=OLED_FONT_MEDIUM
            )

            width = box[2] - box[0]

            draw.text(
                (
                    (OLED_WIDTH - width) // 2,
                    28
                ),
                text,
                font=OLED_FONT_MEDIUM,
                fill=1
            )

        draw.text(
            (29, 52),
            "Please wait...",
            font=OLED_FONT_SMALL,
            fill=1
        )

        oled.display(image)

    except Exception as e:
        print("OLED poweroff error:", e)


# ============================================================
# I2C
# ============================================================

print()
print("=" * 70)
print("INITIALIZING I2C")
print("=" * 70)

try:
    i2c = busio.I2C(
        board.SCL,
        board.SDA
    )

    print("I2C OK")

except Exception as e:
    print("I2C ERROR:", e)
    sys.exit(1)


# ============================================================
# ADS1115
# ============================================================

print()
print("=" * 70)
print("INITIALIZING ADS1115")
print("=" * 70)

try:
    ads = ADS1115(i2c)
    ads.gain = 1

    battery_channel = AnalogIn(
        ads,
        ads1x15.Pin.A1
    )

    print("ADS1115 OK")
    print("Battery input: A1")

except Exception as e:
    print("ADS1115 ERROR:", e)
    sys.exit(1)


# ============================================================
# GPIO
# ============================================================

GPIO.setmode(GPIO.BCM)

GPIO.setup(
    BUTTON_PIN,
    GPIO.IN,
    pull_up_down=GPIO.PUD_UP
)


# ============================================================
# BATTERY PERCENTAGE
# ============================================================

def calculate_battery_percent(voltage):
    if voltage <= BATTERY_EMPTY_VOLTAGE:
        return 0

    if voltage >= BATTERY_FULL_VOLTAGE:
        return 100

    percentage = (
        (voltage - BATTERY_EMPTY_VOLTAGE)
        /
        (BATTERY_FULL_VOLTAGE - BATTERY_EMPTY_VOLTAGE)
    ) * 100.0

    return int(
        round(
            max(0, min(100, percentage))
        )
    )


# ============================================================
# READ BATTERY
# ============================================================

def read_battery():
    global last_battery_voltage
    global last_battery_percent

    try:
        a1_voltage = battery_channel.voltage

        calculated_voltage = (
            a1_voltage * DIVIDER_RATIO
        )

        calibrated_voltage = (
            calculated_voltage * CALIBRATION_FACTOR
        )

        percentage = calculate_battery_percent(
            calibrated_voltage
        )

        last_battery_voltage = calibrated_voltage
        last_battery_percent = percentage

        return calibrated_voltage, percentage

    except Exception as e:
        print("[BATTERY ERROR]", e)
        return None, None


# ============================================================
# STOP AUDIO
# ============================================================

def stop_audio():
    global audio_process

    if audio_process is None:
        return

    try:
        if audio_process.poll() is None:
            audio_process.terminate()

            try:
                audio_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                audio_process.kill()

                try:
                    audio_process.wait(timeout=1)
                except Exception:
                    pass

    except Exception as e:
        print("Audio stop error:", e)

    audio_process = None


# ============================================================
# START AUDIO
# ============================================================

def start_arecord():
    command = [
        "arecord",
        "-D", AUDIO_DEVICE,
        "-f", AUDIO_FORMAT,
        "-r", str(AUDIO_RATE),
        "-c", str(AUDIO_CHANNELS),
        "-t", "raw"
    ]

    print()
    print("Starting INMP441")
    print("Command:", " ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0
    )

    return process


# ============================================================
# START ESP32 SCRIPT
# ============================================================

def start_esp32_script():
    global esp32_process
    global asphyxia_mode

    if asphyxia_mode:
        return

    if not os.path.exists(ESP32_SCRIPT):
        print("ERROR: esp32_f.py not found:")
        print(ESP32_SCRIPT)
        return

    print()
    print("=" * 70)
    print("STARTING ESP32 EMERGENCY MODE")
    print("=" * 70)

    try:
        esp32_process = subprocess.Popen(
            [
                PYTHON_EXECUTABLE,
                ESP32_SCRIPT
            ],
            cwd=BASE_DIR
        )

        asphyxia_mode = True

        print("esp32_f.py started")
        print("PID:", esp32_process.pid)

    except Exception as e:
        print("ESP32 start error:", e)


# ============================================================
# STOP ESP32 SCRIPT
# ============================================================

def stop_esp32_script():
    global esp32_process
    global asphyxia_mode

    if esp32_process is not None:
        print("Stopping esp32_f.py...")

        try:
            if esp32_process.poll() is None:
                esp32_process.terminate()

                try:
                    esp32_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    esp32_process.kill()

                    try:
                        esp32_process.wait(timeout=1)
                    except Exception:
                        pass

        except Exception as e:
            print("ESP32 stop error:", e)

    esp32_process = None
    asphyxia_mode = False


# ============================================================
# ACTUAL LINUX POWER OFF
# ============================================================

def execute_system_poweroff():
    print()
    print("=" * 70)
    print("EXECUTING ACTUAL RASPBERRY PI POWER OFF")
    print("=" * 70)

    try:
        os.sync()
    except Exception as e:
        print("sync error:", e)

    # The service must run as user babypi.
    # /etc/sudoers.d/babypi-poweroff grants these commands
    # without asking for a password.
    commands = [
        ["sudo", "-n", "/usr/bin/systemctl", "poweroff"],
        ["sudo", "-n", "/sbin/poweroff"],
        ["sudo", "-n", "/usr/sbin/poweroff"],
    ]

    for command in commands:
        print("Trying:", " ".join(command))

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )

            print("Return code:", result.returncode)

            if result.stdout.strip():
                print("stdout:", result.stdout.strip())

            if result.stderr.strip():
                print("stderr:", result.stderr.strip())

            if result.returncode == 0:
                print("POWER OFF COMMAND ACCEPTED")
                return True

        except subprocess.TimeoutExpired:
            # A shutdown command may stop returning while the
            # system is already shutting down.
            print("Poweroff command timed out.")
            print("Assuming shutdown is in progress.")
            return True

        except Exception as e:
            print("Poweroff command error:", e)

    print()
    print("=" * 70)
    print("ERROR: ACTUAL POWER OFF FAILED")
    print("=" * 70)

    print(
        "Check /etc/sudoers.d/babypi-poweroff"
    )

    return False


# ============================================================
# POWER OFF
# ============================================================

def power_off():
    global running
    global poweroff_requested
    global shutdown_in_progress

    with shutdown_lock:
        if poweroff_requested:
            print("Poweroff already requested.")
            return

        poweroff_requested = True
        shutdown_in_progress = True

    print()
    print("=" * 70)
    print("POWER OFF REQUESTED")
    print("=" * 70)

    # Stop normal processing.
    running = False

    try:
        stop_esp32_script()
    except Exception as e:
        print("ESP32 shutdown error:", e)

    try:
        stop_audio()
    except Exception as e:
        print("Audio shutdown error:", e)

    # ========================================================
    # EXACT 1 -> 2 -> 3 COUNTDOWN
    # ========================================================
    for number in range(
        1,
        SHUTDOWN_COUNTDOWN_SECONDS + 1
    ):
        print(
            f"POWER OFF COUNTDOWN: {number}"
        )

        show_powering_off(number)

        end_time = (
            time.monotonic()
            +
            SHUTDOWN_STEP_SECONDS
        )

        while time.monotonic() < end_time:
            time.sleep(0.05)

    # Final screen.
    show_powering_off(0)

    time.sleep(0.5)

    # OLED must be hidden BEFORE Linux shutdown.
    oled_off()

    time.sleep(0.5)

    print("OLED is OFF.")
    print("Sending actual Linux poweroff...")

    success = execute_system_poweroff()

    if success:
        print("Linux accepted the poweroff command.")

        # Normally systemd will terminate this process.
        while True:
            time.sleep(1)

    print()
    print("=" * 70)
    print("WARNING: RASPBERRY PI DID NOT POWER OFF")
    print("=" * 70)
    print("Check sudoers configuration.")


# ============================================================
# REQUEST POWER OFF
# ============================================================

def request_poweroff(reason="UNKNOWN"):
    print()
    print("=" * 70)
    print("POWER OFF REQUEST:", reason)
    print("=" * 70)

    with shutdown_lock:
        if poweroff_requested or shutdown_in_progress:
            print("Shutdown already in progress.")
            return

    thread = threading.Thread(
        target=power_off,
        daemon=False
    )

    thread.start()


# ============================================================
# LOW BATTERY POWER OFF
# ============================================================

def low_battery_poweroff(voltage):
    global low_battery_shutdown
    global running

    with shutdown_lock:
        if low_battery_shutdown or poweroff_requested:
            return

        low_battery_shutdown = True

    print()
    print("=" * 70)
    print("LOW BATTERY DETECTED")
    print(f"Battery: {voltage:.3f} V")
    print(f"Limit: {BATTERY_LOW_VOLTAGE:.3f} V")
    print("=" * 70)

    running = False

    try:
        stop_audio()
    except Exception:
        pass

    try:
        stop_esp32_script()
    except Exception:
        pass

    # Show LOW BATTERY for exactly approximately 5 seconds.
    show_low_battery(voltage)

    end_time = (
        time.monotonic()
        +
        LOW_BATTERY_DISPLAY_SECONDS
    )

    while time.monotonic() < end_time:
        time.sleep(0.05)

    request_poweroff("LOW BATTERY")


# ============================================================
# BATTERY MONITOR
# ============================================================

def battery_monitor():
    global running

    print()
    print("=" * 70)
    print("BATTERY MONITOR STARTED")
    print("=" * 70)

    while running:
        if shutdown_in_progress:
            return

        try:
            voltage, percentage = read_battery()

            if voltage is not None:
                print(
                    f"[BATTERY] "
                    f"{voltage:.3f} V "
                    f"{percentage}%"
                )

                if voltage <= BATTERY_LOW_VOLTAGE:
                    low_battery_poweroff(voltage)
                    return

        except Exception as e:
            print("Battery monitor error:", e)

        # Check every BATTERY_CHECK_INTERVAL seconds.
        # Use small sleeps so shutdown is noticed quickly.
        for _ in range(20):
            if not running or shutdown_in_progress:
                return

            time.sleep(
                BATTERY_CHECK_INTERVAL / 20.0
            )


# ============================================================
# LOAD MODEL FILES
# ============================================================

print()
print("=" * 70)
print("LOADING ML MODEL")
print("=" * 70)

for required_file in (
    MODEL_TFLITE,
    LABEL_MAP_PATH,
    NORMALIZATION_PATH
):
    if not os.path.exists(required_file):
        raise FileNotFoundError(required_file)


# ============================================================
# LABEL MAP
# ============================================================

with open(
    LABEL_MAP_PATH,
    "r",
    encoding="utf-8"
) as f:
    label_map = json.load(f)

CLASSES = {
    int(k): v
    for k, v in label_map.items()
}

print()
print("Classes:")

for index in sorted(CLASSES):
    print(
        f"  {index}: {CLASSES[index]}"
    )


# ============================================================
# NORMALIZATION
# ============================================================

with open(
    NORMALIZATION_PATH,
    "r",
    encoding="utf-8"
) as f:
    normalization = json.load(f)

SAMPLE_RATE = int(
    normalization["sample_rate"]
)

N_MFCC = int(
    normalization["n_mfcc"]
)

N_FFT = int(
    normalization["n_fft"]
)

WIN_LENGTH = float(
    normalization["win_length"]
)

HOP_LENGTH = float(
    normalization["hop_length"]
)

MAX_FRAMES = int(
    normalization["max_frames"]
)

MEAN = np.asarray(
    normalization["mean"],
    dtype=np.float32
).reshape(
    1,
    N_MFCC
)

STD = np.asarray(
    normalization["std"],
    dtype=np.float32
).reshape(
    1,
    N_MFCC
)

STD = np.maximum(
    STD,
    1e-6
)

print("Sample rate:", SAMPLE_RATE)
print("MFCC:", N_MFCC)
print("N_FFT:", N_FFT)
print("Win length:", WIN_LENGTH)
print("Hop length:", HOP_LENGTH)
print("Max frames:", MAX_FRAMES)


# ============================================================
# LOAD TFLITE
# ============================================================

print()
print("Loading TFLite...")

interpreter = tflite.Interpreter(
    model_path=MODEL_TFLITE,
    num_threads=4
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded")
print("Input:", input_details[0]["shape"])
print("Output:", output_details[0]["shape"])


# ============================================================
# MFCC
# ============================================================

def extract_mfcc_from_audio(audio):
    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if len(audio) == 0:
        raise ValueError("Empty audio")

    audio = audio - np.mean(audio)

    peak = np.max(np.abs(audio))

    if peak > 1e-8:
        audio = audio / peak

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=int(
            SAMPLE_RATE * HOP_LENGTH
        ),
        win_length=int(
            SAMPLE_RATE * WIN_LENGTH
        ),
        window="hann",
        center=True
    )

    mfcc = mfcc.T

    original_frames = mfcc.shape[0]

    if original_frames < MAX_FRAMES:
        pad_amount = MAX_FRAMES - original_frames

        mfcc = np.pad(
            mfcc,
            (
                (0, pad_amount),
                (0, 0)
            ),
            mode="constant"
        )
    else:
        mfcc = mfcc[:MAX_FRAMES, :]

    mfcc = (mfcc - MEAN) / STD

    mfcc = np.expand_dims(
        mfcc,
        axis=-1
    )

    mfcc = np.expand_dims(
        mfcc,
        axis=0
    )

    return mfcc.astype(np.float32)


# ============================================================
# AUDIO GAIN
# ============================================================

def apply_audio_gain(audio):
    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    audio = audio * AUDIO_GAIN

    audio = np.clip(
        audio,
        -1.0,
        1.0
    )

    return audio


# ============================================================
# PREDICT
# ============================================================

def predict_audio(audio):
    global last_prediction
    global last_confidence

    with prediction_lock:
        audio = apply_audio_gain(audio)

        features = extract_mfcc_from_audio(audio)

        expected_shape = tuple(
            input_details[0]["shape"]
        )

        actual_shape = tuple(
            features.shape
        )

        if expected_shape != actual_shape:
            interpreter.resize_tensor_input(
                input_details[0]["index"],
                actual_shape,
                strict=False
            )

            interpreter.allocate_tensors()

            input_details[:] = (
                interpreter.get_input_details()
            )

            output_details[:] = (
                interpreter.get_output_details()
            )

        input_dtype = input_details[0]["dtype"]

        features = features.astype(
            input_dtype
        )

        start = time.perf_counter()

        interpreter.set_tensor(
            input_details[0]["index"],
            features
        )

        interpreter.invoke()

        inference_time = (
            time.perf_counter() - start
        )

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )

        output = output[0]

        quantization = output_details[0].get(
            "quantization",
            (0.0, 0)
        )

        scale, zero_point = quantization

        if (
            output_details[0]["dtype"] != np.float32
            and scale != 0
        ):
            output = (
                output.astype(np.float32)
                - zero_point
            ) * scale

        output = np.asarray(
            output,
            dtype=np.float32
        )

        # Some models output logits rather than probabilities.
        # If values are not already a probability distribution,
        # softmax is safer than simply dividing by their sum.
        if (
            np.all(output >= 0)
            and
            np.isfinite(output).all()
            and
            abs(float(np.sum(output)) - 1.0) < 0.05
        ):
            probabilities = output
        else:
            shifted = output - np.max(output)
            exp_values = np.exp(shifted)

            denominator = np.sum(exp_values)

            if denominator <= 0:
                probabilities = np.zeros_like(output)
            else:
                probabilities = (
                    exp_values / denominator
                )

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = CLASSES.get(
            predicted_index,
            f"Unknown {predicted_index}"
        )

        confidence = float(
            probabilities[predicted_index]
        )

        last_prediction = predicted_class
        last_confidence = confidence

        print()
        print("=" * 70)
        print("PREDICTION")
        print("Condition:", predicted_class)
        print(
            "Confidence:",
            f"{confidence * 100:.2f}%"
        )
        print(
            "Inference:",
            f"{inference_time:.3f}s"
        )
        print("=" * 70)

        return predicted_class, confidence


# ============================================================
# RECORDING
# ============================================================

TARGET_SAMPLES = int(
    RECORD_SECONDS * AUDIO_RATE
)

PRE_TRIGGER_SAMPLES = int(
    PRE_TRIGGER_SECONDS * AUDIO_RATE
)

pre_buffer = collections.deque(
    maxlen=PRE_TRIGGER_SAMPLES
)


# ============================================================
# PCM S32 -> FLOAT
# ============================================================

def pcm_s32_to_float(raw):
    if not raw:
        return np.array(
            [],
            dtype=np.float32
        )

    count = len(raw) // 4

    values = np.frombuffer(
        raw[:count * 4],
        dtype="<i4"
    ).astype(
        np.float32
    )

    values /= 2147483648.0

    return values


# ============================================================
# AUDIO LEVEL
# ============================================================

def audio_level(samples):
    if len(samples) == 0:
        return 0.0, 0.0

    samples = np.asarray(
        samples,
        dtype=np.float32
    )

    rms = float(
        np.sqrt(
            np.mean(
                samples * samples
            )
        )
    )

    peak = float(
        np.max(
            np.abs(samples)
        )
    )

    return rms, peak


# ============================================================
# RECORD 4 SECONDS
# ============================================================

def record_triggered_audio(process, first_samples):
    print()
    print("=" * 70)
    print("HIGH SOUND DETECTED")
    print("RECORDING 4 SECONDS")
    print("GAIN = 6x")
    print("=" * 70)

    show_recording()

    collected = []

    # Include the samples which caused the trigger.
    collected.extend(
        first_samples.tolist()
    )

    while len(collected) < TARGET_SAMPLES:
        if not running:
            break

        raw = process.stdout.read(4096)

        if not raw:
            break

        samples = pcm_s32_to_float(raw)

        if len(samples) == 0:
            continue

        collected.extend(
            samples.tolist()
        )

    collected = collected[:TARGET_SAMPLES]

    audio = np.asarray(
        collected,
        dtype=np.float32
    )

    print(
        "Recorded samples:",
        len(audio)
    )

    print(
        "Recorded seconds:",
        f"{len(audio) / AUDIO_RATE:.2f}"
    )

    return audio


# ============================================================
# COMPLETE PROGRAM RESTART
# ============================================================

def complete_reset():
    global running
    global reset_in_progress

    if reset_in_progress:
        return

    reset_in_progress = True

    print()
    print("=" * 70)
    print("SINGLE PRESS")
    print("COMPLETE PROGRAM RESTART")
    print("=" * 70)

    running = False

    stop_esp32_script()
    stop_audio()

    # Clear OLED before replacing this process.
    oled_off()

    time.sleep(0.2)

    try:
        # IMPORTANT:
        # If this program is launched by systemd using the venv
        # interpreter, sys.executable remains:
        #
        # /home/babypi/baby_model/.venv/bin/python
        #
        # Therefore the restarted program stays inside the venv.
        os.execv(
            sys.executable,
            [
                sys.executable,
                os.path.abspath(__file__)
            ]
        )

    except Exception as e:
        print("Restart error:", e)
        os._exit(1)


# ============================================================
# BUTTON THREAD
# ============================================================

def button_thread():
    global running

    press_count = 0
    last_press_time = 0

    print()
    print("=" * 70)
    print("BUTTON MONITOR")
    print("=" * 70)
    print("GPIO14")
    print("Single = PROGRAM RESTART")
    print("Double = BATTERY")
    print(
        f"Long = POWER OFF "
        f"(hold {LONG_PRESS_TIME:.1f}s)"
    )
    print("=" * 70)

    while running:
        if shutdown_in_progress:
            return

        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            press_start = time.monotonic()

            time.sleep(DEBOUNCE_TIME)

            if GPIO.input(BUTTON_PIN) != GPIO.LOW:
                continue

            long_press = False

            while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                elapsed = (
                    time.monotonic()
                    -
                    press_start
                )

                if elapsed >= LONG_PRESS_TIME:
                    long_press = True

                    print()
                    print("=" * 70)
                    print("LONG PRESS DETECTED")
                    print(
                        f"Held for {elapsed:.2f} seconds"
                    )
                    print(
                        "STARTING 1 -> 2 -> 3 "
                        "POWER OFF SEQUENCE"
                    )
                    print(
                        "6 seconds total"
                    )
                    print("=" * 70)

                    request_poweroff(
                        "GPIO14 LONG PRESS"
                    )

                    return

                time.sleep(0.01)

            if not long_press:
                press_count += 1
                last_press_time = time.monotonic()

        if press_count > 0:
            if (
                time.monotonic()
                -
                last_press_time
                >
                MULTI_PRESS_TIME
            ):
                if press_count == 1:
                    print("SINGLE PRESS")
                    complete_reset()
                    return

                elif press_count == 2:
                    print("DOUBLE PRESS")

                    voltage = last_battery_voltage
                    percentage = last_battery_percent

                    # If battery has not been sampled yet,
                    # take a fresh reading.
                    if voltage <= 0:
                        new_voltage, new_percentage = (
                            read_battery()
                        )

                        if new_voltage is not None:
                            voltage = new_voltage
                            percentage = new_percentage

                    print(
                        f"Battery: "
                        f"{voltage:.2f} V "
                        f"{percentage}%"
                    )

                    show_battery(
                        voltage,
                        percentage
                    )

                    # Do not block shutdown indefinitely.
                    end_time = (
                        time.monotonic()
                        + 2.0
                    )

                    while time.monotonic() < end_time:
                        if shutdown_in_progress:
                            return

                        time.sleep(0.05)

                    if shutdown_in_progress:
                        return

                    if asphyxia_mode:
                        show_asphyxia()

                    elif processing_prediction:
                        show_processing()

                    else:
                        show_waiting()

                press_count = 0

        time.sleep(0.01)


# ============================================================
# START BATTERY THREAD
# ============================================================

battery_thread = threading.Thread(
    target=battery_monitor,
    daemon=True
)

battery_thread.start()


# ============================================================
# START BUTTON THREAD
# ============================================================

button_monitor = threading.Thread(
    target=button_thread,
    daemon=True
)

button_monitor.start()


# ============================================================
# STOP BOOT OLED
# ============================================================
#
# Do this before the normal monitor screen.
# ============================================================

stop_boot_oled_service()


# ============================================================
# INITIAL OLED
# ============================================================

show_waiting()


# ============================================================
# MAIN
# ============================================================

try:
    print()
    print("=" * 70)
    print("BABY MONITOR STARTED")
    print("=" * 70)

    print("Python:", sys.executable)
    print("Base:", BASE_DIR)

    print("INMP441:", AUDIO_DEVICE)
    print("Audio rate:", AUDIO_RATE)
    print("Audio gain:", AUDIO_GAIN)
    print("Recording:", RECORD_SECONDS, "seconds")
    print("RMS threshold:", SOUND_RMS_THRESHOLD)
    print("Peak threshold:", SOUND_PEAK_THRESHOLD)

    print(
        "Battery low:",
        BATTERY_LOW_VOLTAGE,
        "V"
    )

    print(
        "Low battery display:",
        LOW_BATTERY_DISPLAY_SECONDS,
        "seconds"
    )

    print(
        "Shutdown countdown:",
        "1 -> 2 -> 3"
    )

    print(
        "Shutdown duration:",
        "6 seconds"
    )

    print("=" * 70)

    audio_process = start_arecord()

    last_trigger = 0.0

    while running:
        if shutdown_in_progress:
            break

        # ====================================================
        # ASPHYXIA MODE
        # ====================================================

        if asphyxia_mode:
            if esp32_process is not None:
                if esp32_process.poll() is not None:
                    print(
                        "WARNING: esp32_f.py exited."
                    )

                    esp32_process = None

                    # Keep emergency state on OLED.
                    asphyxia_mode = True
                    show_asphyxia()

            time.sleep(0.1)
            continue

        # ====================================================
        # AUDIO PROCESS RECOVERY
        # ====================================================

        if audio_process is None:
            audio_process = start_arecord()
            time.sleep(0.1)
            continue

        # ====================================================
        # AUDIO READ
        # ====================================================

        raw = audio_process.stdout.read(4096)

        if not raw:
            print("Audio stream stopped.")

            stop_audio()

            time.sleep(0.5)

            if running:
                audio_process = start_arecord()

            continue

        samples = pcm_s32_to_float(raw)

        if len(samples) == 0:
            continue

        # ====================================================
        # SOUND LEVEL
        # ====================================================

        rms, peak = audio_level(samples)

        for sample in samples:
            pre_buffer.append(float(sample))

        # ====================================================
        # SOUND TRIGGER
        # ====================================================

        high_sound = (
            rms >= SOUND_RMS_THRESHOLD
            or
            peak >= SOUND_PEAK_THRESHOLD
        )

        now = time.monotonic()

        if (
            now - last_trigger
            <
            DETECTION_COOLDOWN
        ):
            continue

        # ====================================================
        # HIGH SOUND
        # ====================================================

        if high_sound:
            last_trigger = now

            print()
            print("HIGH AUDIO DETECTED")
            print(f"RMS  = {rms:.4f}")
            print(f"PEAK = {peak:.4f}")

            audio = record_triggered_audio(
                audio_process,
                samples
            )

            if len(audio) == 0:
                show_waiting()
                continue

            # =================================================
            # PROCESSING
            # =================================================

            processing_prediction = True
            show_processing()

            print()
            print("PROCESSING AUDIO...")

            try:
                prediction, confidence = (
                    predict_audio(audio)
                )

                show_prediction(
                    prediction,
                    confidence
                )

                processing_prediction = False

                # =================================================
                # ASPHYXIA
                # =================================================

                if (
                    prediction.strip().lower()
                    ==
                    "asphyxia"
                ):
                    print()
                    print("!" * 70)
                    print("ASPHYXIA DETECTED")
                    print(
                        f"Confidence: "
                        f"{confidence * 100:.2f}%"
                    )
                    print("STARTING ESP32 SCRIPT")
                    print("!" * 70)

                    show_asphyxia()

                    start_esp32_script()

                    continue

                # =================================================
                # NORMAL PREDICTION
                # =================================================

                print()
                print(
                    f"Prediction: {prediction}"
                )

                print(
                    f"Confidence: "
                    f"{confidence * 100:.2f}%"
                )

                print(
                    f"Showing prediction for "
                    f"{PREDICTION_DISPLAY_SECONDS:.0f} seconds..."
                )

                end_time = (
                    time.monotonic()
                    +
                    PREDICTION_DISPLAY_SECONDS
                )

                while time.monotonic() < end_time:
                    if not running:
                        break

                    if shutdown_in_progress:
                        break

                    time.sleep(0.1)

                if running:
                    print(
                        "Returning to waiting for cry..."
                    )

                    show_waiting()

            except Exception as e:
                processing_prediction = False

                print()
                print("=" * 70)
                print("MODEL ERROR")
                print(e)
                print("=" * 70)

                show_waiting()


except KeyboardInterrupt:
    print()
    print("CTRL+C pressed")


except Exception as e:
    print()
    print("=" * 70)
    print("MAIN PROGRAM ERROR")
    print("=" * 70)
    print(e)
    print("=" * 70)


finally:
    running = False

    print()
    print("Cleaning up...")

    try:
        stop_esp32_script()
    except Exception:
        pass

    try:
        stop_audio()
    except Exception:
        pass

    # If shutdown is already executing, power_off() is
    # responsible for OLED shutdown.
    if not shutdown_in_progress:
        try:
            oled_off()
        except Exception:
            pass

    try:
        GPIO.cleanup()
    except Exception:
        pass

    print("System stopped.")
