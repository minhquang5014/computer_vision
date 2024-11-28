import cv2
import numpy as np
from pymodbus.client import ModbusTcpClient as mbc
import time
from tkinter import *
from PIL import Image, ImageTk

class CameraApp(Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera App")
        self.cap = cv2.VideoCapture(0)
        self.width = 640
        self.height = 480
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.video_label = Label(self)
        self.video_label.pack()

        # Modbus client initialization
        self.client = mbc('192.168.0.1', port=502)
        if self.client.connect():
            print('Connected to PLC')
        else:
            print('Failed to connect to PLC')
            exit()

        self.UNIT = 0x1  # rack, slot

        # Define color ranges in HSV
        self.lower_red = np.array([0, 150, 150])
        self.upper_red = np.array([12, 255, 255])
        self.lower_yellow = np.array([18, 120, 120])
        self.upper_yellow = np.array([35, 255, 255])
        self.lower_blue = np.array([90, 120, 120])
        self.upper_blue = np.array([115, 255, 255])
        self.lower_purple = np.array([130, 80, 80])
        self.upper_purple = np.array([165, 255, 255])

        # Initialize register states and timing for non-blocking delay
        self.register_states = [0, 0, 0, 0]
        self.last_update_time = time.time()
        self.update_interval = 1  # Update interval in seconds

        # Initialize Modbus registers
        for i in range(4):
            self.write_register(i, 0)

        # Start video loop
        self.video_loop()

    def write_register(self, register, value):
        response = self.client.write_register(register, value, unit=self.UNIT)
        if response.isError():
            print(f'Error writing to register {register}: {response}')
        else:
            print(f'Successfully wrote {value} to register {register}')

    def video_loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.width, self.height))
            self.fg_mask = self.background_subtractor.apply(frame)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert BGR image to HSV format

            masks = [
                (cv2.inRange(img, self.lower_red, self.upper_red), 0, "red Object"),
                (cv2.inRange(img, self.lower_yellow, self.upper_yellow), 1, "yellow Object"),
                (cv2.inRange(img, self.lower_blue, self.upper_blue), 2, "blue Object"),
                (cv2.inRange(img, self.lower_purple, self.upper_purple), 3, "purple Object")
            ]

            self.register_states = [0, 0, 0, 0]

            for mask, register, label in masks:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 3000:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (76, 153, 0), 3)  # Draw rectangle
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        self.register_states[register] = 1
                        break

            if time.time() - self.last_update_time >= self.update_interval:
                for i in range(4):
                    self.write_register(i, self.register_states[i])
                self.last_update_time = time.time()

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.after(10, self.video_loop)  # Update the frame every 10 ms

    def __del__(self):
        # Release the video capture when the object is destroyed
        if self.cap.isOpened():
            self.cap.release()
        self.client.close()  # Close Modbus client connection

if __name__ == "__main__":
    app = CameraApp()
    app.mainloop()