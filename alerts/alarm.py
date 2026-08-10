from playsound import playsound
import threading
import os

alarm_running = False


def _play():

    global alarm_running

    alarm_running = True

    try:

        alarm_file = os.path.join(

            os.path.dirname(__file__),

            "alarm.wav"

        )

        playsound(alarm_file)

    except Exception as e:

        print()

        print("Alarm Error")

        print(e)

    finally:

        alarm_running = False


def trigger_alarm():

    global alarm_running

    if alarm_running:

        return

    threading.Thread(

        target=_play,

        daemon=True

    ).start()