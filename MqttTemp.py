import DHT11 as dht
import time
import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
import json
import base64

client = mqtt.Client()
client.connect("broker.hivemq.com", 1883, 60)

def loop():
    while(True):
        with open("apfel.jpg", "rb") as image2string:
            image_data = image2string.read()

        humidity, temperature = dht.getTemperature()
        
        fruits = {
            "apple": 'middle',
            "banana": 'left'
            #"image": base64.b64encode(image_data).decode('utf-8')
            }
            
        dictionary = { 
            "temperature": temperature,
            "humidity": humidity,
            "image": base64.b64encode(image_data).decode('utf-8')
        }

        json_file = json.dumps(fruits)
        
        publish.single("Smarthome/Innsbruck/temperature", json_file, hostname="broker.hivemq.com")
       
        print("Sent data to Server")
        time.sleep(5)

if __name__ == '__main__':
    print ('Program is starting ... ')
    try:
        loop()
    except KeyboardInterrupt:
        exit()
