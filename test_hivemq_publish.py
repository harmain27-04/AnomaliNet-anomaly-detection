import paho.mqtt.client as mqtt
import ssl
import time


MQTT_HOST = "b5581c5800a543359895176cc38fba67.s1.eu.hivemq.cloud"
MQTT_PORT = 8883

MQTT_USERNAME = "AnomaliNet"
MQTT_PASSWORD = "Harmain@27"

MQTT_TOPIC = "anomalinet/test"


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.username_pw_set(
    MQTT_USERNAME,
    MQTT_PASSWORD
)

client.tls_set(
    cert_reqs=ssl.CERT_REQUIRED
)

print("=" * 60)
print("HIVEMQ CLOUD PUBLISHER TEST")
print("=" * 60)

print("Connecting to:")
print(MQTT_HOST)

client.connect(
    MQTT_HOST,
    MQTT_PORT,
    keepalive=60
)

print("CONNECTED TO HIVEMQ CLOUD")

client.loop_start()

time.sleep(1)

result = client.publish(
    MQTT_TOPIC,
    "HELLO FROM ANOMALINET AI LAPTOP",
    qos=1
)

result.wait_for_publish()

print("MESSAGE PUBLISHED")
print("Topic:", MQTT_TOPIC)
print("Message: HELLO FROM ANOMALINET AI LAPTOP")

time.sleep(2)

client.loop_stop()
client.disconnect()

print("DISCONNECTED")
print("=" * 60)