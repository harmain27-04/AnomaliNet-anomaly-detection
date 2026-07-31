from alerts.alarm import trigger_alarm
from alerts.snapshot_capture import save_snapshot

from alerts.email_alert import send_email
from alerts.telegram_alert import send_telegram


def trigger_all_alerts(frame=None):

    snapshot = None

    print("=" * 70)
    print("Triggering Alerts")
    print("=" * 70)

    if frame is not None:

        snapshot = save_snapshot(frame)

    try:

        trigger_alarm()

    except Exception as e:

        print("Alarm Error :", e)

    try:

        if snapshot:

            send_email(snapshot)

    except Exception as e:

        print("Email Error :", e)

    try:

        if snapshot:

            send_telegram(snapshot)

    except Exception as e:

        print("Telegram Error :", e)

    return snapshot