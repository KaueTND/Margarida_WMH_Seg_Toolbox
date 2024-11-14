from customtkinter import CTkButton, CTkLabel, CTkEntry, CTkCheckBox
from customtkinter import *
from tkinter import filedialog, Toplevel,Tk, StringVar
import tkinter as tk
from PIL import Image
class Settings:
    def __init__(self, master,path_dire):
        #super().__init__(master)
        self.master = master
        self.master.title("Settings")
        self.master.iconbitmap(path_dire+'softwareLogo.ico')
        self.master.configure(bg="#4C52E9")#a=
        self.master.geometry("700x300")
        self.path_dir = path_dire#'C:/Users/kaueu/Desktop/customtkinter-examples-master/WMH_app/'
        
        #self.architecture = arch
        #self.orientation = orient
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
        #pass
        # # First Line: Input Directory
        #logo_img_data = Image.open(self.path_dir+"brainLogo.png")
        #logo_img = CTkImage(dark_image=logo_img_data, light_image=logo_img_data, size=(77.68, 85.42))

                
        self.add_label(self.path_dir+"ants.png","   ANTS N4BiasFieldCorrection ",self.input_dir_var)
        self.add_label(self.path_dir+"rotate.png","   FSL Reorient2Std",self.output_dir_var)
        
        analytics_img_data = Image.open(self.path_dir+"run.png")
        analytics_img = CTkImage(dark_image=analytics_img_data, light_image=analytics_img_data)
        CTkButton(master=self.master, image=analytics_img, text="Accept", fg_color="#878CFA", font=("Arial Bold", 14), hover_color="#474BB5", anchor="w",command=self.run_multiple_patients).pack(anchor="center", ipady=5, pady=(16, 0))

        # # Run Button
        # CTkButton(self, text="Run Multiple Patients", command=self.run_multiple_patients).grid(row=4, column=0, columnspan=3, pady=10)

    def browse_dir(self,variable):
        input_dir = filedialog.askdirectory()
        if input_dir:
            variable.set(input_dir)

    def run_multiple_patients(self):
        print("Running")
        self.master.destroy()
        
        #print(self.get_output())
    
    @staticmethod
    def show_settings():
        root = Toplevel()  # Create a new top-level window
        app = Settings(root)  
        return [app.input_dir_var.get(),app.output_dir_var.get()]

# Example of how to use the class
if __name__ == "__main__":
    root = CTk()
    app = Settings(root)
    root.mainloop()
