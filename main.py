import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import customtkinter
import sqlite3

def resize_image(img, scale_percent) :
    # Calculate new size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # Resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def draw_box(img, result, class_list):
    # Get information from result
    xyxy= result.boxes.xyxy.numpy()
    confidence= result.boxes.conf.numpy()
    class_id= result.boxes.cls.numpy().astype(int)
    # Get Class name
    class_name = [class_list[x] for x in class_id]
    # Pack together for easy use
    sum_output = list(zip(class_name, confidence,xyxy))   
    # Copy image, in case that we need original image for something
    out_image = img.copy()
    
    for run_output in sum_output :
        # Unpack
        label, con, box = run_output
        print (label)
        
        # Choose color  
        box_color = (0, 0, 255)
        text_color = (255,255,255)
        # Draw object box
        first_half_box = (int(box[0]),int(box[1]))
        second_half_box = (int(box[2]),int(box[3]))
        cv2.rectangle(out_image, first_half_box, second_half_box, box_color, 2)
        # Create text
        text_print = '{label} {con:.2f}'.format(label = label, con = con)
        # Locate text position
        text_location = (int(box[0]), int(box[1] - 10 ))
        # Get size and baseline
        labelSize, baseLine = cv2.getTextSize(text_print, cv2.FONT_HERSHEY_SIMPLEX, 1, 2) 
        # Draw text's background
        cv2.rectangle(out_image 
                        , (int(box[0]), int(box[1] - labelSize[1] - 10 ))
                        , (int(box[0])+labelSize[0], int(box[1] + baseLine-10))
                        , box_color , cv2.FILLED) 
        # Put text
        cv2.putText(out_image, text_print ,text_location
                    , cv2.FONT_HERSHEY_SIMPLEX , 1
                    , text_color, 2 ,cv2.LINE_AA)
    return out_image

class GuiYoloV8:
    def __init__(self, parent=None):
        self.run_model = False
        self.window = customtkinter.CTk(parent)
        self.window.title('Car seats Detection')
        self.window.geometry("700x450")
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=1)

        # Create Layout of the GUI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        self.navigation_frame = customtkinter.CTkFrame(self.window, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame_title = customtkinter.CTkLabel(self.navigation_frame, text="Barcode Confirmation",
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_title.grid(row=0, column=0, padx=10, pady=10)
        
        # Create class detection frame  
        self.navigation_frame_title = customtkinter.CTkLabel(self.navigation_frame, text="Class Detected",
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_title.grid(row=2, column=0, padx=10, pady=10)

        # create home frame
        self.home_frame = customtkinter.CTkFrame(self.window, corner_radius=0, fg_color="transparent")
        self.home_frame.grid(row=0, column=1, sticky="nsew")
        self.home_frame.grid_columnconfigure(0, weight=1)
        
        # Create form for model and parameter input
        customtkinter.CTkLabel(self.home_frame , text='Enter Model Name', width=15).grid(row=0, column=0, pady=0)
        self.model_name_input = customtkinter.CTkEntry(self.home_frame , textvariable=customtkinter.StringVar(), width=100)
        self.model_name_input.insert(customtkinter.END, 'best.pt')
        self.model_name_input.grid(row=0, column=1, pady=10)
        
        # Create form for Camera scale
        customtkinter.CTkLabel(self.home_frame , text='Scale to Show', width=15).grid(row=1, column=0, pady=0)
        self.scale_percent_input = customtkinter.CTkEntry(self.home_frame , textvariable=customtkinter.StringVar(), width=100)
        self.scale_percent_input.insert(customtkinter.END, '50')
        self.scale_percent_input.grid(row=1, column=1, pady=10)
        
        # Create form for Barcode
        customtkinter.CTkLabel(self.home_frame , text='Enter Barcode', width=15).grid(row=2, column=0, pady=0)
        self.barcode_input = customtkinter.CTkEntry(self.home_frame , textvariable=customtkinter.StringVar(), width=100)
        self.barcode_input.insert(customtkinter.END, '?')
        self.barcode_input.grid(row=2, column=1, pady=10)

        # Button for Run, Stop, Close, Enter
        customtkinter.CTkButton(self.home_frame , text='Run', command=self.start_running).grid(row=4, column=0, padx=10, pady=10)
        customtkinter.CTkButton(self.home_frame , text='Stop', command=self.stop_running).grid(row=4, column=1, padx=10, pady=10)
        customtkinter.CTkButton(self.home_frame , text='Close', command=self.close_gui).grid(row=4, column=2, padx=10, pady=10)
        customtkinter.CTkButton(self.home_frame , text='Enter', command=self.status_check).grid(row=2, column=2, padx=10, pady=10)

        # Image label for Camera config
        self.image_label = tk.Label(self.home_frame, bg='#242424')
        self.image_label.grid(row=5,  column=0, padx=10, pady=10)
        
        #Create result label
        self.result_barcode = tk.Label(self.navigation_frame, text="")
        self.result_barcode.grid(row=1, column=0)
        
        #Create result classes
        self.result_classes = tk.Label(self.navigation_frame, text="")
        self.result_classes.grid(row=3, column=0)
        
    def start_running(self):
        self.run_model = True
        # Set up model and parameter
        model = YOLO(self.model_name_input.get())
        class_list = model.model.names
        scale_show = int(self.scale_percent_input.get())
        # Read Video
        video = cv2.VideoCapture(0)
        while self.run_model:
            ret, frame = video.read()
            if ret:
                results = model.predict(frame)
                labeled_img = draw_box(frame, results[0], class_list)
                # Retrieve detected classes
                detected_classes = self.get_detected_classes(results, class_list)  
                display_img = resize_image(labeled_img, scale_show)
                rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_img)
                # Show Image
                photo_img = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo_img)
                self.image_label.photo = photo_img
                
                self.window.update()
                
                # Update the label with detected classes
                self.result_classes.config(text="Detected Classes: " + ", ".join(detected_classes))
                
            else:
                # Break the loop if not read
                video.release()
                self.run_model
                
    def get_barcode(self):
        return str(self.barcode_input.get())  
                
    def get_detected_classes(self, results, class_list):
        self.detected_classes = []
        xyxy = results[0].boxes.xyxy.numpy()
        confidence = results[0].boxes.conf.numpy()
        class_id = results[0].boxes.cls.numpy().astype(int)
        class_name = [class_list[x] for x in class_id]
        sum_output = list(zip(class_name, confidence, xyxy))
        
        for run_output in sum_output:
            label, con, box = run_output
            self.detected_classes.append(label)
            
        return self.detected_classes
    
    def status_check(self):
        conn = sqlite3.connect('carseat.db')
        c = conn.cursor()
        
        barcode = self.get_barcode()
        detected_classes = self.result_classes.cget("text").replace("Detected Classes: ", "").split(", ")
        # Convert list to string
        detected_classes = ", ".join(detected_classes)  
        
        c.execute("SELECT * FROM data WHERE code = ? AND class = ?", (barcode, detected_classes))

        rowcount = len(c.fetchall())

        if rowcount >= 1:
            # Update the database or perform other actions for a match
            print("MATCH")
            self.result_barcode.config(text="Match!")
            
        else:
            # Handle the case for mismatch
            print("MISMATCH")
            self.result_barcode.config(text="Mismatch!")
            
        print(rowcount)
          
    def stop_running(self):
        self.run_model = False

    def close_gui(self):
        self.window.destroy()

gui_yolo_v8 = GuiYoloV8()
gui_yolo_v8.window.mainloop()
