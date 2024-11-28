import cv2
import tkinter as tk
from tkinter import Button
from PIL import ImageTk, Image
import threading
import datetime
import os
import face_recognition
import cvzone

class Webcam:
    def __init__(self, window, window_title):
        self.window = window
        self.window_title = window_title

        self.video_capture = 1
        self.vid = cv2.VideoCapture(self.video_capture)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        self.btn_record = Button(window, text="record", width=15, command=self.record)
        self.btn_record.pack(side=tk.LEFT)

        self.btn_snapphoto = Button(window, text="take photo", width=15, command = self.take_photo)
        self.btn_snapphoto.pack(side=tk.LEFT)
    
        self.show_box = False
        self.text = tk.Text(window, width = int(self.width/40), height = int(self.height / 200))
        self.text.tag_configure("center", justify='center', foreground="red", font=("helvetica", 12, "bold"))
        self.text.insert(tk.END, "no face detected")
        self.text.tag_add("center", "1.0", "end")
        self.text.configure(state=tk.DISABLED)
        self.text.pack(side=tk.LEFT)

        self.button_show = Button(window, text="Show_bounding_box", width = 20, command = self.toggle_box)
        self.button_show.pack(side=tk.LEFT)

        self.is_recording = False
        self.out = None
        
        self.directories = "images_and_video"
        if not os.path.exists(self.directories):
            os.makedirs(self.directories)
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
        
    def toggle_box(self):
        self.show_box = not self.show_box

    def face_detected(self):
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "face_detected")
        self.text.tag_add("center", "1.0", "end")
        self.text.configure(state=tk.DISABLED)
    def no_face_detected(self):
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "no face detected")
        self.text.tag_add("center", "1.0", "end")
        self.text.configure(state=tk.DISABLED)

    def take_photo(self):
        ret, frame = self.vid.read()
        if ret:
            filename = os.path.join(self.directories,f"photo_{datetime.datetime.now().strftime('%d%m_%Hh%Mm%Ss')}.jpg")
            cv2.imwrite(filename, frame)
            print(f"photo saved as {filename}")

    def record(self):
        if self.is_recording:
            self.is_recording = False
            if self.out:
                self.out.release()
                self.out = None
            self.btn_record.config(text="Start Recording")
            print("Recording stopped")
        else:
            self.is_recording = True
            file_name = os.path.join(self.directories, f"video_{datetime.datetime.now().strftime('%d%M%Y_%Hh%Mm%Ss')}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.out = cv2.VideoWriter(file_name, fourcc, 20.0, (frame_width, frame_height))
            self.btn_record.config(text="Stop recording")
            print(f"Recording started. Saving to {file_name}")
        
    def update(self):
        ret, frame = self.vid.read()
        frame = cv2.flip(frame, 1)
        if ret:
            if self.is_recording and self.out is not None:
                self.out.write(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frameS = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
            frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)

            faceLocs = face_recognition.face_locations(frameS)

            if faceLocs:
                self.face_detected()
                if self.show_box:
                    for faceLoc in faceLocs:
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
                        bbox = x1, y1, x2 - x1, y2 - y1
                        cvzone.cornerRect(frame, bbox, rt=0)
            else:
                self.no_face_detected()
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update)   
    def on_closing(self):
        if self.is_recording:
            self.is_recording = False
            if self.out is not None:
                self.out.release()
                self.out = None
        self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = Webcam(root, "Webcam App")