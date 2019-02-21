# app.py
import picamera
import time

camera = picamera.PiCamera()
camera.start_preview()
time.sleep(5)
camera.capture('snapshot.jpg')
camera.stop_preview()
