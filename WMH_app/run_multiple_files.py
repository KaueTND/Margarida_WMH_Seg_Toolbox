from customtkinter import CTkButton, CTkLabel, CTkEntry, CTkCheckBox, CTkTextbox
from customtkinter import *
from tkinter import filedialog, Toplevel,Tk, StringVar
import tkinter as tk
from PIL import Image
from models import Models

class MultPatients:
    def __init__(self, master, path_model,path_dire):
        #super().__init__(master)
        self.master = master
        self.master.title("Run Multiple Patients")
        self.master.iconbitmap('softwareLogo.ico')
        self.master.configure(bg="#4C52E9")#a=
        self.master.geometry("700x500")
        self.path_dir = path_dire
        self.text_widget = None
        #self.architecture = arch
        #self.orientation = orient
        self.path_model = path_model
        # Variables to store directory paths
        self.export_dir_var = tk.StringVar()
        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()

        # Create and place widgets
        self.create_widgets()
        
    def add_label(self,icon_name,text,variable):
        gridtitleLong = CTkFrame(master=self.master, fg_color="transparent")
        gridtitleLong.pack(fill="both", padx=10, pady=(0,0))
        gridtitleLong.columnconfigure(0, weight=4)
        gridtitleLong.columnconfigure(1, weight=20)
        gridtitleLong.columnconfigure(2, weight=2)
        
        email_icon_data = Image.open(icon_name)
        email_icon = CTkImage(dark_image=email_icon_data, light_image=email_icon_data, size=(20,20))
        CTkLabel(master=gridtitleLong, text=text,image=email_icon, anchor="w", justify="left", font=("Arial", 12), text_color="#ffffff", fg_color="transparent", compound="left").grid(row=0, column=0, sticky=tk.W+tk.E, pady=(15,0))
        CTkEntry(master=gridtitleLong, textvariable=variable, width=10, justify="left", font=("Arial", 12)).grid(row=0, column=1, sticky=tk.W+tk.E, pady=(15,0))
        analytics_img_data = Image.open(self.path_dir+"search.png")
        analytics_img = CTkImage(dark_image=analytics_img_data, light_image=analytics_img_data)
        CTkButton(master=gridtitleLong, image=analytics_img, text="", font=("Arial", 12), fg_color="transparent",width=1,command=lambda: self.browse_dir(variable)).grid(row=0, column=2, sticky=tk.W+tk.E, pady=(15,0))
    
    def create_widgets(self):
                        
        self.add_label(self.path_dir+"upload.png","   Input Dir* ",self.input_dir_var)
        self.add_label(self.path_dir+"download.png","   Output Dir*",self.output_dir_var)
        self.add_label(self.path_dir+"export_pdf.png","   Export PDF",self.export_dir_var)

        # # Checkboxes in a 1x3 grid
        gridtitleLong = CTkFrame(master=self.master, fg_color="transparent")
        gridtitleLong.pack(fill="both", padx=10, pady=(0,0))
        gridtitleLong.columnconfigure(0, weight=4)
        gridtitleLong.columnconfigure(1, weight=20)
        gridtitleLong.columnconfigure(2, weight=2)
        
        CTkCheckBox(master=gridtitleLong,corner_radius=50, fg_color="#474BB5",hover_color="#474BB5", text="Preprocess").grid(row=3, column=0, padx=5, pady=(20,5))
        CTkCheckBox(master=gridtitleLong,corner_radius=50, fg_color="#474BB5",hover_color="#474BB5",text="Execute").grid(row=3, column=1, padx=5, pady=5)
        CTkCheckBox(master=gridtitleLong,corner_radius=50, fg_color="#474BB5",hover_color="#474BB5",text="Calculate").grid(row=3, column=2, padx=5, pady=5)

        analytics_img_data = Image.open(self.path_dir+"run.png")
        analytics_img = CTkImage(dark_image=analytics_img_data, light_image=analytics_img_data)
        CTkButton(master=self.master, image=analytics_img, text="Generate Masks", fg_color="#878CFA", font=("Arial Bold", 14), hover_color="#474BB5", anchor="w",command=self.run_multiple_patients).pack(anchor="center", ipady=5, pady=(16, 0))
        CTkLabel(master=self.master, text="Output").pack(pady=(0, 0), anchor="center")
        self.text_widget = CTkTextbox(self.master,height=200)
        self.text_widget.pack(fill="both", expand=True)
        # # Run Button
        # CTkButton(self, text="Run Multiple Patients", command=self.run_multiple_patients).grid(row=4, column=0, columnspan=3, pady=10)

    def browse_dir(self,variable):
        input_dir = filedialog.askdirectory()
        if input_dir:
            variable.set(input_dir)

    def update_text_widget(self, text):
        # Update the CTkTextbox with the captured output
        self.text_widget.insert("end", text)
        self.text_widget.see("end")
        
    def run_multiple_patients(self):
        print("Running")
        #print(self.get_output())
        model = Models(self.path_model, 
                       self.output_dir_var.get(),
                       path_in=self.input_dir_var.get(),
                       path_pdf=self.export_dir_var.get(),
                       text_widget=self.text_widget)
        model.run_multiple_pipeline()

    @staticmethod
    def show_custom_tkinter_screen(out_path,path_dire):
        root = Toplevel()  # Create a new top-level window
        app = MultPatients(root,out_path,path_dire)  

# Example of how to use the class
if __name__ == "__main__":
    root = CTk()
    app = MultPatients(root)
    root.mainloop()
