import os
import winsound
import threading

alarm_running = False


def _play():

    global alarm_running

    alarm_running = True

    try:

        alarm_path = os.path.abspath(

            os.path.join(

                os.path.dirname(__file__),

                "alarm.wav"

            )

        )

        if not os.path.exists(alarm_path):

            print("Alarm file not found")

            return

        print("Playing Alarm...")

        winsound.PlaySound(

            alarm_path,

            winsound.SND_FILENAME

        )

    except Exception as e:

        print("Alarm Error :", e)

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