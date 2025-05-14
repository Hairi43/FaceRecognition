import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import cv2
import threading
from live_face import LiveFace
import copy

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facio Recognitio")
        self.root.grid()
                

        # grid 3x1
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.columnconfigure(3, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=1)


        self.source_label = ttk.Label(root, text="Wybierz źródło obrazu:")
        self.source_label.grid(row=1, column=0, sticky="sw", padx=20, pady=10)

        self.option_label = ttk.Label(root, text="Dodatkowe opcje:")
        self.option_label.grid(row=2, column=0, sticky="sw", padx=20, pady=10)


        self.choose_btn = ttk.Button(root, text="Z pliku wideo", command=self.choose_file)
        self.choose_btn.grid(row=2, column=0, sticky="n")

        self.live_btn = ttk.Button(root, text="Na żywo", command=self.live_video)
        self.live_btn.grid(row=2, column=1, sticky="nw")


        # configurator
        self.crop = tk.BooleanVar()
        self.faceDet = tk.BooleanVar()
        self.faceLand = tk.BooleanVar()
        self.gamma_corr = tk.DoubleVar()

        config_frame = ttk.Frame(root)
        config_frame.grid(row=3, column=0, sticky="nw", padx=20)

        self.config_label = ttk.Label(config_frame, text="Konfigurator")
        self.config_label.pack(anchor="w")

        c1 = ttk.Checkbutton(config_frame, text='Kadrowanie', variable=self.crop, onvalue=True, offvalue=False)
        c2 = ttk.Checkbutton(config_frame, text='Obrys twarzy', variable=self.faceDet, onvalue=True, offvalue=False)
        c3 = ttk.Checkbutton(config_frame, text='Cechy twarzy', variable=self.faceLand, onvalue=True, offvalue=False)
        passw_label = tk.Label(config_frame, text = 'Korekcja gamma, [0.0, 1.0]')
        c4 = ttk.Entry(config_frame, text='Korekcja gamma', textvariable=self.gamma_corr)

        c1.pack(anchor='w')
        c2.pack(anchor='w')
        c3.pack(anchor='w')
        passw_label.pack(anchor='w')
        c4.pack(anchor='w')


        self.record_btn = ttk.Button(root, text="Rozpocznij nagrywanie", command=self.start_recording)
        self.record_btn.grid(row=3, column=1, sticky="nw")
        self.img_btn = ttk.Button(root, text="Zrob zdjecie", command=self.start_picture)
        self.img_btn.grid(row=3, column=2, sticky="nw")

        self.video_source = 0
        self.from_file = 0
        self.cap = None
        self.recording = False
        self.picture = False
        self.output_file = "recordingsDB/output.mp4"
        self.img_output_file = "imagesDB/output.jpg"

    def choose_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pliki wideo", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_source = file_path
            # uruchomienie rozpoznawania twarzy
            live_face = LiveFace(self.video_source, draw_crop=self.crop.get(), draw_face=self.faceDet.get(),
                                  draw_landmarks=self.faceLand.get(), gamma_corr=self.gamma_corr.get())
            live_face.run()


    # def open_configurator(self):
    #     messagebox.showinfo("Konfigurator", "Tutaj pojawi się konfigurator (do zaimplementowania).")

    def start_recording(self):
        self.recording = True
        threading.Thread(target=self.record_video, daemon=True).start()
    
    def start_picture(self):
        self.picture = True
        threading.Thread(target=self.get_picture, daemon=True).start()


    # kadrowanie
    def draw_lines(self, frame):
        height, width = frame.shape[:2]

        x1 = int(0.35 * width)
        x2 = int(0.65 * width)
        y1 = int(0.2 * height)
        y2 = int(0.8 * height)

        start_point_v1 = (x1, 0)
        end_point_v1 = (x1, height)
        start_point_v2 = (x2, 0)
        end_point_v2 = (x2, height)

        start_point_h1 = (0, y1)
        end_point_h1 = (width, y1)
        start_point_h2 = (0, y2)
        end_point_h2 = (width, y2)

        color = (0, 0, 255)
        thickness = 4

        frame_ = copy.deepcopy(frame)

        frame_ = cv2.line(frame_, start_point_v1, end_point_v1, color, thickness)
        frame_ = cv2.line(frame_, start_point_v2, end_point_v2, color, thickness)
        frame_ = cv2.line(frame_, start_point_h1, end_point_h1, color, thickness)
        frame_ = cv2.line(frame_, start_point_h2, end_point_h2, color, thickness)

        return frame_



    def record_video(self):
        cam = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.output_file, fourcc, 20.0, (frame_width, frame_height))

        while cam.isOpened() and self.recording:
            ret, frame = cam.read()
            if not ret:
                break

            if cv2.waitKey(1) == ord('q'):
                break

            if self.crop.get():
                frame_lines = self.draw_lines(frame)

            out.write(frame)
            cv2.imshow('Nagrywanie', frame_lines)

        cam.release()
        out.release()
        cv2.destroyAllWindows()
        self.recording = False
    


    def get_picture(self):
        print("Start funkcji get_picture")
        cam = cv2.VideoCapture(0)

        if not cam.isOpened():
            messagebox.showerror("Błąd kamery", "Nie można otworzyć kamery.")
            return

        while cam.isOpened() and self.picture:
            ret, frame = cam.read()
            print("ret:", ret)

            if not ret or frame is None:
                print("Błąd pobierania obrazu z kamery.")
                continue 
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                cv2.imwrite(self.img_output_file, frame)
                print(f"Zapisano zdjęcie: {self.img_output_file}")
                break
            elif key == ord('q'):
                print("Zamknięcie bez zapisu.")
                break

            if self.crop.get():
                frame = self.draw_lines(frame)

            cv2.imshow("Podglad zdjecia (E = zapisz, Q = wyjdz)", frame)

        cam.release()
        cv2.destroyAllWindows()
        self.picture = False

    
    def live_video(self):
        # if self.video_source == 0:
        #     return
        # live_face = LiveFace(self.video_source)
        # live_face.run()
        live_face = LiveFace(self.video_source, draw_crop=self.crop.get(), draw_face=self.faceDet.get(),
                                  draw_landmarks=self.faceLand.get(), gamma_corr=self.gamma_corr.get())
        live_face.run_live()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.geometry("640x480")
    root.mainloop()
