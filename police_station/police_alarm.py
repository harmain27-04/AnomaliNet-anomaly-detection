"""
=========================================================
AnomaliNet Police Station Alarm
=========================================================
Plays a local alarm whenever a new anomaly incident
is received by the Police Workstation.
=========================================================
"""

import os
import threading
from playsound import playsound


# =====================================================
# ALARM STATE
# =====================================================

alarm_running = False


# =====================================================
# PLAY ALARM
# =====================================================

def _play_alarm():

    global alarm_running

    alarm_running = True

    try:

        alarm_file = os.path.join(
            os.path.dirname(__file__),
            "police_alarm.wav"
        )

        if not os.path.exists(alarm_file):

            print(
                "Police alarm file not found:"
            )

            print(
                alarm_file
            )

            return

        print()
        print("=" * 70)
        print("🚨🚨🚨 POLICE ALARM TRIGGERED 🚨🚨🚨")
        print("=" * 70)

        playsound(
            alarm_file
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("POLICE ALARM ERROR")
        print(type(e).__name__)
        print(e)
        print("=" * 70)

    finally:

        alarm_running = False


# =====================================================
# TRIGGER POLICE ALARM
# =====================================================

def trigger_police_alarm():

    global alarm_running

    if alarm_running:

        print(
            "Police alarm is already playing."
        )

        return

    threading.Thread(
        target=_play_alarm,
        daemon=True
    ).start()