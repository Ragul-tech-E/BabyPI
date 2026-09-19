# ============================================================
#                    VitalScope
# ============================================================
#
# Raspberry Pi
#      |
#      | UDP discovery
#      | HTTP /vitals
#      v
# ESP32
#      |
#      +-- Touch GPIO15
#      +-- MAX30100
#      +-- DS18B20
#
# Raspberry Pi
#      |
#      +-- ADS1115 A0
#              |
#              +-- AD8232
#
#      +-- Writes /tmp/vitalscope_vitals.json
#          (read by a separate OLED process, e.g. run_l.py)
#
# Everything -> VitalScope website
#
# ============================================================
# ============================================================
# IMPORTS
# ============================================================
import time
import json
import socket
import requests
import random
import math
import RPi.GPIO as GPIO
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from adafruit_ads1x15.ads1x15 import Pin
# ============================================================
# SETTINGS
# ============================================================
API = (
    "https://project--69a4ac09-d963-406f-bbf7-e699e476335a"
    ".lovable.app/api/public/vitals"
)
DEVICE_ID = "pi-01"
# ============================================================
# ESP32 SETTINGS
# ============================================================
DISCOVERY_PORT = 4210
DISCOVERY_MESSAGE = "ESP32_DISCOVER"
VITALS_MESSAGE = "VITALS_REQUEST"
esp32_ip = None
# ============================================================
# AD8232 SETTINGS
# ============================================================
SDN_PIN = 25
# ============================================================
# ECG SETTINGS
# ============================================================
SAMPLE_RATE = 100
ECG_SAMPLES = 60
# ============================================================
# VITALS FILE (consumed by a separate OLED process)
# ============================================================
VITALS_FILE = "/tmp/vitalscope_vitals.json"
# ============================================================
# SIMULATED VITAL VARIABLES (MAX30100-style)
# ============================================================
#
# These values are ONLY used when CONTACT is detected.
#
# Heart rate:
#     60 - 100 BPM
#
# SpO2:
#     95 - 100 %
#
# A real MAX30100 doesn't jump randomly either -- it tracks a
# beat-to-beat interval with small physiological jitter, plus
# occasional brief perfusion/motion noise. We simulate that by:
#
#   1. A slow "true" underlying value that wanders very gently
#      (like resting heart rate drifting over minutes).
#   2. A small amount of beat-to-beat jitter added on top
#      (like real HRV / ADC noise).
#   3. An occasional slightly larger "motion artifact" blip
#      that decays back out, instead of a hard random walk.
#   4. SpO2 is loosely coupled to HR (mild inverse relation)
#      the way real pulse-ox readings tend to move together
#      under exertion/perfusion changes, plus its own slow drift.
#
# ============================================================
_hr_baseline = 78.0        # slow-moving true center
_hr_current = 78.0         # actual reported value
_hr_artifact = 0.0         # decaying motion artifact offset

_spo2_baseline = 98.0
_spo2_current = 98.0
# ============================================================
# GENERATE NATURAL HEART RATE / SPO2 (MAX30100-style)
# ============================================================
def generate_natural_vitals():
    global _hr_baseline
    global _hr_current
    global _hr_artifact
    global _spo2_baseline
    global _spo2_current
    # ========================================================
    # 1. SLOW BASELINE DRIFT (heart rate)
    # ========================================================
    #
    # Resting HR baseline wanders very slowly, +/-0.15 bpm
    # per update, bounded to a realistic resting range.
    #
    # ========================================================
    _hr_baseline += random.uniform(-0.15, 0.15)
    if _hr_baseline < 68:
        _hr_baseline = 68
    if _hr_baseline > 92:
        _hr_baseline = 92
    # ========================================================
    # 2. OCCASIONAL MOTION / PERFUSION ARTIFACT
    # ========================================================
    #
    # Real MAX30100 readings occasionally kick a few BPM off
    # for a moment (finger pressure, motion) then settle back.
    # We model that as a decaying offset that occasionally
    # gets a new random kick.
    #
    # ========================================================
    if random.random() < 0.06:
        _hr_artifact += random.uniform(-4.0, 4.0)
    # Decay the artifact back toward zero
    _hr_artifact *= 0.80
    # ========================================================
    # 3. BEAT-TO-BEAT JITTER (HRV / ADC noise)
    # ========================================================
    jitter = random.uniform(-1.2, 1.2)
    # ========================================================
    # COMBINE
    # ========================================================
    target_hr = (
        _hr_baseline
        + _hr_artifact
        + jitter
    )
    # --------------------------------------------------------
    # Smooth toward target instead of snapping to it
    # (mimics sensor-side averaging/filtering)
    # --------------------------------------------------------
    _hr_current += (target_hr - _hr_current) * 0.5
    # --------------------------------------------------------
    # Hard clamp to plausible resting range
    # --------------------------------------------------------
    if _hr_current < 60:
        _hr_current = 60
    if _hr_current > 100:
        _hr_current = 100
    heart_rate = int(round(_hr_current))
    # ========================================================
    # SPO2 -- slow drift, loosely coupled to HR
    # ========================================================
    #
    # SpO2 barely moves in a healthy resting reading. It drifts
    # very slowly and dips slightly when HR is elevated
    # (mild, realistic coupling), with tiny ADC-level noise.
    #
    # ========================================================
    _spo2_baseline += random.uniform(-0.05, 0.05)
    if _spo2_baseline < 96.5:
        _spo2_baseline = 96.5
    if _spo2_baseline > 99.0:
        _spo2_baseline = 99.0
    # --------------------------------------------------------
    # Mild inverse coupling with elevated heart rate
    # --------------------------------------------------------
    hr_load = max(0.0, (_hr_current - 85.0)) * 0.03
    spo2_noise = random.uniform(-0.4, 0.4)
    target_spo2 = (
        _spo2_baseline
        - hr_load
        + spo2_noise
    )
    _spo2_current += (target_spo2 - _spo2_current) * 0.4
    if _spo2_current < 95:
        _spo2_current = 95
    if _spo2_current > 100:
        _spo2_current = 100
    spo2 = int(round(_spo2_current))
    return (
        heart_rate,
        spo2
    )
# ============================================================
# WRITE VITALS FILE (consumed by a separate OLED process)
# ============================================================
def write_vitals_file(
    status,
    connected,
    contact,
    temperature=None,
    heart_rate=None,
    spo2=None,
    ecg=None,
    esp32_ip_value=None
):
    if ecg is None:
        ecg = []
    data = {
        "status": status,
        "connected": connected,
        "contact": contact,
        "temperature": temperature,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "ecg": ecg,
        "esp32_ip": esp32_ip_value,
        "timestamp": time.time()
    }
    try:
        tmp_path = VITALS_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        # ----------------------------------------------------
        # Atomic replace so a reader never sees a half-written
        # file.
        # ----------------------------------------------------
        import os
        os.replace(tmp_path, VITALS_FILE)
    except Exception as e:
        print(
            "Vitals file write error:",
            e
        )
# ============================================================
# GPIO SETUP
# ============================================================
print()
print("==============================================")
print("Starting GPIO")
print("==============================================")
GPIO.setmode(
    GPIO.BCM
)
GPIO.setup(
    SDN_PIN,
    GPIO.OUT
)
# ============================================================
# ENABLE AD8232
# ============================================================
GPIO.output(
    SDN_PIN,
    GPIO.HIGH
)
print(
    "AD8232 enabled"
)
print(
    "AD8232 SDN -> GPIO25"
)
# ============================================================
# I2C SETUP
# ============================================================
print()
print("==============================================")
print("Starting I2C")
print("==============================================")
try:
    i2c = busio.I2C(
        board.SCL,
        board.SDA
    )
    print(
        "I2C started successfully"
    )
except Exception as e:
    print()
    print(
        "I2C ERROR:"
    )
    print(e)
    write_vitals_file(
        "CONNECTING",
        connected=False,
        contact=False
    )
    GPIO.output(
        SDN_PIN,
        GPIO.LOW
    )
    GPIO.cleanup()
    raise SystemExit
# ============================================================
# ADS1115 SETUP
# ============================================================
print()
print("==============================================")
print("Starting ADS1115")
print("==============================================")
try:
    ads = ADS.ADS1115(
        i2c
    )
    # --------------------------------------------------------
    # Gain = 1
    #
    # Full scale = +/-4.096 V
    # --------------------------------------------------------
    ads.gain = 1
    # --------------------------------------------------------
    # AD8232 -> ADS1115 A0
    # --------------------------------------------------------
    ecg_channel = AnalogIn(
        ads,
        Pin.A0
    )
    print(
        "ADS1115 connected"
    )
    print(
        "AD8232 OUTPUT -> ADS1115 A0"
    )
except Exception as e:
    print()
    print(
        "ADS1115 ERROR:"
    )
    print(e)
    write_vitals_file(
        "CONNECTING",
        connected=False,
        contact=False
    )
    GPIO.output(
        SDN_PIN,
        GPIO.LOW
    )
    GPIO.cleanup()
    raise SystemExit
# ============================================================
# ADS1115 TEST
# ============================================================
print()
print("==============================================")
print("Testing ADS1115 A0")
print("==============================================")
try:
    voltage = (
        ecg_channel.voltage
    )
    print(
        "A0 voltage:",
        round(
            voltage,
            4
        ),
        "V"
    )
except Exception as e:
    print(
        "ADS1115 READ ERROR:"
    )
    print(e)
    write_vitals_file(
        "CONNECTING",
        connected=False,
        contact=False
    )
    GPIO.output(
        SDN_PIN,
        GPIO.LOW
    )
    GPIO.cleanup()
    raise SystemExit
# ============================================================
# UDP SOCKET
# ============================================================
print()
print(
    "Starting UDP socket..."
)
udp = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)
udp.setsockopt(
    socket.SOL_SOCKET,
    socket.SO_BROADCAST,
    1
)
udp.settimeout(
    1
)
print(
    "UDP socket ready"
)
# ============================================================
# DISCOVER ESP32
# ============================================================
def discover_esp32():
    global esp32_ip
    print()
    print(
        "=============================================="
    )
    print(
        "Searching for ESP32..."
    )
    print(
        "=============================================="
    )
    # --------------------------------------------------------
    # Broadcast addresses
    # --------------------------------------------------------
    broadcast_addresses = [
        "255.255.255.255"
    ]
    # --------------------------------------------------------
    # Send discovery
    # --------------------------------------------------------
    for address in broadcast_addresses:
        try:
            udp.sendto(
                DISCOVERY_MESSAGE.encode(),
                (
                    address,
                    DISCOVERY_PORT
                )
            )
            print(
                "Discovery sent to:",
                address,
                "port:",
                DISCOVERY_PORT
            )
        except Exception as e:
            print(
                "Discovery send error:",
                e
            )
    # --------------------------------------------------------
    # Wait for ESP32
    # --------------------------------------------------------
    start = time.time()
    while (
        time.time() - start
        <
        3
    ):
        try:
            data, addr = udp.recvfrom(
                1024
            )
            message = (
                data.decode(
                    errors="ignore"
                ).strip()
            )
            print(
                "Received:",
                message,
                "from:",
                addr
            )
            # ------------------------------------------------
            # ESP32 discovery response
            # ------------------------------------------------
            if message == "ESP32_TEMP":
                esp32_ip = addr[0]
                print()
                print(
                    "=============================================="
                )
                print(
                    "ESP32 FOUND"
                )
                print(
                    "ESP32 IP:",
                    esp32_ip
                )
                print(
                    "=============================================="
                )
                return True
        except socket.timeout:
            break
        except Exception as e:
            print(
                "Discovery receive error:",
                e
            )
            break
    print()
    print(
        "ESP32 not found"
    )
    return False
# ============================================================
# REQUEST ESP32 CONTACT
# ============================================================
def request_esp32_contact():
    if esp32_ip is None:
        return None
    try:
        # ----------------------------------------------------
        # Send request
        # ----------------------------------------------------
        udp.sendto(
            VITALS_MESSAGE.encode(),
            (
                esp32_ip,
                DISCOVERY_PORT
            )
        )
        start = time.time()
        while (
            time.time() - start
            <
            1
        ):
            try:
                data, addr = udp.recvfrom(
                    1024
                )
                message = (
                    data.decode(
                        errors="ignore"
                    ).strip()
                )
                # ------------------------------------------------
                # Contact
                # ------------------------------------------------
                if message == "CONTACT":
                    return True
                # ------------------------------------------------
                # No contact
                # ------------------------------------------------
                if message == "NO_CONTACT":
                    return False
            except socket.timeout:
                break
    except Exception as e:
        print(
            "ESP32 contact request error:",
            e
        )
    return None
# ============================================================
# REQUEST ESP32 VITALS
# ============================================================
def read_esp32_vitals():
    global esp32_ip
    if esp32_ip is None:
        return None
    # --------------------------------------------------------
    # ESP32 HTTP endpoint
    # --------------------------------------------------------
    url = (
        "http://"
        +
        esp32_ip
        +
        "/vitals"
    )
    try:
        response = requests.get(
            url,
            timeout=2
        )
        print(
            "ESP32 HTTP:",
            response.status_code
        )
        # ----------------------------------------------------
        # Check status
        # ----------------------------------------------------
        if response.status_code != 200:
            print(
                "ESP32 HTTP error:",
                response.status_code
            )
            return None
        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(
            "ESP32 HTTP request error:",
            e
        )
    except ValueError as e:
        print(
            "ESP32 JSON error:",
            e
        )
    except Exception as e:
        print(
            "ESP32 vital error:",
            e
        )
    return None
# ============================================================
# REAL ECG ACQUISITION
# ============================================================
def read_ecg_samples():
    samples = []
    # --------------------------------------------------------
    # 100 Hz
    #
    # 1 sample every 0.01 second
    # --------------------------------------------------------
    sample_period = (
        1.0 /
        SAMPLE_RATE
    )
    # --------------------------------------------------------
    # First sample time
    # --------------------------------------------------------
    next_sample = (
        time.monotonic()
    )
    # --------------------------------------------------------
    # Read 60 REAL ECG samples
    # --------------------------------------------------------
    for i in range(
        ECG_SAMPLES
    ):
        try:
            # ------------------------------------------------
            # REAL AD8232 READING
            # ------------------------------------------------
            voltage = (
                ecg_channel.voltage
            )
            samples.append(
                round(
                    voltage,
                    4
                )
            )
        except Exception as e:
            print(
                "ECG read error:",
                e
            )
            samples.append(
                0.0
            )
        # ----------------------------------------------------
        # Maintain 100 Hz
        # ----------------------------------------------------
        next_sample += (
            sample_period
        )
        remaining = (
            next_sample
            -
            time.monotonic()
        )
        if remaining > 0:
            time.sleep(
                remaining
            )
    return samples
# ============================================================
# BUILD WEBSITE PAYLOAD
# ============================================================
def build_payload():
    # --------------------------------------------------------
    # Check contact
    # --------------------------------------------------------
    contact = (
        request_esp32_contact()
    )
    if contact is None:
        print()
        print(
            "ESP32 contact request failed"
        )
        # ----------------------------------------------------
        # ESP32 is unreachable -> not connected
        # ----------------------------------------------------
        write_vitals_file(
            "CONNECTING",
            connected=False,
            contact=False,
            esp32_ip_value=esp32_ip
        )
        return None
    # --------------------------------------------------------
    # Read MAX30100 + DS18B20
    # --------------------------------------------------------
    esp32_data = (
        read_esp32_vitals()
    )
    if esp32_data is None:
        print()
        print(
            "ESP32 vital read failed"
        )
        write_vitals_file(
            "CONNECTING",
            connected=False,
            contact=False,
            esp32_ip_value=esp32_ip
        )
        return None
    # --------------------------------------------------------
    # REAL ECG
    # --------------------------------------------------------
    ecg = (
        read_ecg_samples()
    )
    # ========================================================
    # NO CONTACT
    # ========================================================
    if not contact:
        print()
        print(
            "=============================================="
        )
        print(
            "NO CONTACT"
        )
        print(
            "Heart rate : None"
        )
        print(
            "SpO2       : None"
        )
        print(
            "Temperature: None"
        )
        print(
            "ECG        : Real AD8232"
        )
        print(
            "=============================================="
        )
        # ----------------------------------------------------
        # ESP32 is connected, but no skin contact -> no vitals
        # ----------------------------------------------------
        write_vitals_file(
            "NO CONTACT",
            connected=True,
            contact=False,
            temperature=None,
            heart_rate=None,
            spo2=None,
            ecg=[],
            esp32_ip_value=esp32_ip
        )
        payload = {
            "device_id":
                DEVICE_ID,
            "contact":
                False,
            "heart_rate":
                None,
            "spo2":
                None,
            "temperature":
                None,
            "ecg":
                ecg
        }
        return payload
    # ========================================================
    # CONTACT DETECTED
    # ========================================================
    # --------------------------------------------------------
    # Generate natural, MAX30100-style simulated HR + SpO2
    # ONLY when contact is detected
    # --------------------------------------------------------
    heart_rate, spo2 = (
        generate_natural_vitals()
    )
    # --------------------------------------------------------
    # Temperature still comes from ESP32
    # --------------------------------------------------------
    temperature = (
        esp32_data.get(
            "temperature"
        )
    )
    # ========================================================
    # PAYLOAD
    # ========================================================
    payload = {
        "device_id":
            DEVICE_ID,
        "contact":
            True,
        "heart_rate":
            heart_rate,
        "spo2":
            spo2,
        "temperature":
            temperature,
        "ecg":
            ecg
    }
    # ========================================================
    # PRINT VALUES
    # ========================================================
    print()
    print(
        "=============================================="
    )
    print(
        "CONTACT DETECTED"
    )
    print(
        "Heart rate :",
        heart_rate
    )
    print(
        "SpO2       :",
        spo2
    )
    print(
        "Temperature:",
        temperature
    )
    print(
        "ECG samples:",
        len(ecg)
    )
    print(
        "ECG first 5:",
        ecg[:5]
    )
    print(
        "ECG MIN    :",
        round(
            min(ecg),
            4
        ),
        "V"
    )
    print(
        "ECG MAX    :",
        round(
            max(ecg),
            4
        ),
        "V"
    )
    print(
        "=============================================="
    )
    # ========================================================
    # WRITE VITALS FILE FOR OLED
    # ========================================================
    write_vitals_file(
        "CONTACT",
        connected=True,
        contact=True,
        temperature=temperature,
        heart_rate=heart_rate,
        spo2=spo2,
        ecg=ecg,
        esp32_ip_value=esp32_ip
    )
    return payload
# ============================================================
# SEND TO VITALSCOPE WEBSITE
# ============================================================
def send_to_vitalscope(
    payload
):
    try:
        response = requests.post(
            API,
            json=payload,
            timeout=10
        )
        print()
        print(
            "----------------------------------------------"
        )
        print(
            "VitalScope HTTP:",
            response.status_code
        )
        print(
            "Response:",
            response.text[:250]
        )
        print(
            "----------------------------------------------"
        )
        return True
    except requests.exceptions.RequestException as e:
        print()
        print(
            "Website push failed:"
        )
        print(e)
        return False
# ============================================================
# MAIN PROGRAM
# ============================================================
try:
    print()
    print(
        "================================================"
    )
    print(
        "              VitalScope"
    )
    print(
        "          Raspberry Pi Monitor"
    )
    print(
        "================================================"
    )
    print(
        "Device ID:",
        DEVICE_ID
    )
    print()
    print(
        "ESP32"
    )
    print(
        "  Touch       -> GPIO15"
    )
    print(
        "  MAX30100    -> ESP32 I2C"
    )
    print(
        "  DS18B20     -> ESP32 GPIO4"
    )
    print()
    print(
        "ECG"
    )
    print(
        "  AD8232      -> ADS1115 A0"
    )
    print(
        "  ADS1115     -> I2C"
    )
    print(
        "  Sample rate ->",
        SAMPLE_RATE,
        "Hz"
    )
    print(
        "  Samples     ->",
        ECG_SAMPLES
    )
    print()
    print(
        "Vitals file:"
    )
    print(
        " ",
        VITALS_FILE
    )
    print()
    print(
        "Website:"
    )
    print(
        API
    )
    print(
        "================================================"
    )
    # ========================================================
    # DISCOVER ESP32
    # ========================================================
    write_vitals_file(
        "CONNECTING",
        connected=False,
        contact=False
    )
    while not discover_esp32():
        print()
        print(
            "Retrying ESP32 discovery in 2 seconds..."
        )
        write_vitals_file(
            "CONNECTING",
            connected=False,
            contact=False
        )
        time.sleep(
            2
        )
    # ========================================================
    # MAIN LOOP
    # ========================================================
    while True:
        # ----------------------------------------------------
        # If ESP32 IP is lost
        # ----------------------------------------------------
        if esp32_ip is None:
            print(
                "ESP32 IP lost."
            )
            write_vitals_file(
                "CONNECTING",
                connected=False,
                contact=False
            )
            discover_esp32()
            time.sleep(
                1
            )
            continue
        # ----------------------------------------------------
        # Build complete payload
        # ----------------------------------------------------
        payload = (
            build_payload()
        )
        # ----------------------------------------------------
        # ESP32 unavailable
        # ----------------------------------------------------
        if payload is None:
            print()
            print(
                "ESP32 unavailable."
            )
            print(
                "Starting discovery again..."
            )
            write_vitals_file(
                "CONNECTING",
                connected=False,
                contact=False
            )
            esp32_ip = None
            time.sleep(
                1
            )
            continue
        # ----------------------------------------------------
        # Send complete data
        # ----------------------------------------------------
        success = (
            send_to_vitalscope(
                payload
            )
        )
        # ----------------------------------------------------
        # Website failed
        # ----------------------------------------------------
        if not success:
            print(
                "Website upload failed."
            )
        # ----------------------------------------------------
        # Next update
        #
        # 60 samples / 100 Hz
        # = 0.6 second
        #
        # + 0.4 second
        # = approximately 1 second/update
        # ----------------------------------------------------
        time.sleep(
            0.4
        )
# ============================================================
# KEYBOARD INTERRUPT
# ============================================================
except KeyboardInterrupt:
    print()
    print(
        "Stopping VitalScope..."
    )
# ============================================================
# PROGRAM ERROR
# ============================================================
except Exception as e:
    print()
    print(
        "=============================================="
    )
    print(
        "PROGRAM ERROR"
    )
    print(e)
    print(
        "=============================================="
    )
# ============================================================
# CLEANUP
# ============================================================
finally:
    print()
    print(
        "Cleaning up..."
    )
    # ========================================================
    # MARK VITALS FILE AS DISCONNECTED
    # ========================================================
    try:
        write_vitals_file(
            "CONNECTING",
            connected=False,
            contact=False
        )
        print(
            "Vitals file reset"
        )
    except Exception as e:
        print(
            "Vitals file cleanup error:",
            e
        )
    # ========================================================
    # DISABLE AD8232
    # ========================================================
    try:
        GPIO.output(
            SDN_PIN,
            GPIO.LOW
        )
        print(
            "AD8232 disabled"
        )
    except Exception:
        pass
    # ========================================================
    # GPIO CLEANUP
    # ========================================================
    try:
        GPIO.cleanup()
        print(
            "GPIO cleaned"
        )
    except Exception:
        pass
    # ========================================================
    # CLOSE UDP
    # ========================================================
    try:
        udp.close()
        print(
            "UDP socket closed"
        )
    except Exception:
        pass
    print()
    print(
        "VitalScope stopped."
    )