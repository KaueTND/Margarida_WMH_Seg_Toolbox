from customtkinter import *
import customtkinter as ctk
from CTkTable import CTkTable
import tkinter as tk
from tkinter import ttk
from PIL import Image
import tkinter
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#from data import *
import numpy as np
import matplotlib
import nibabel as nib
from PIL import Image, ImageTk
from info import *
from tkinter import messagebox, Toplevel
from scipy.ndimage import zoom, rotate
import threading
from run_multiple_files import *
from settings import *
from models import Models
from pdf_exporter import PDFExporter
import json
import pydicom
from metrics import Metrics

class MargaridaWMHSeg:
    def __init__(self):
        self.app = CTk()
        self.app.iconbitmap('softwareLogo.ico')
        self.app.geometry("1600x700")
        self.app.title("Margarida - WMH Segmentation Toolbox")
        
        #Variables of the system
        self.filepath = ''
        self.filepathMask = ''
        self.path_out = ''
        self.status_var = tkinter.IntVar(value=0)
        self.technique_var = tkinter.StringVar(value='Traditional U-Net')
        self.texture_var = tkinter.StringVar(value='GLCM')
        self.architecture_var = tkinter.StringVar(value='VGG16')
        self.cross_lon_var = tkinter.IntVar(value=0)
        self.path_dir = 'C:/Users/kaueu/OneDrive/Área de Trabalho/WMH_app/'
        
        self.frames = [tkinter.PhotoImage(file='loading.gif', format='gif -index %i'%(i)) for i in range(25)]
        self.threshold_var = tkinter.DoubleVar(value=0.5)
        self.popup = ''
        self.sliderThresh = None
        self.sliceAxi_var = tkinter.IntVar(value=200)
        self.sliceCor_var = tkinter.IntVar(value=200)
        self.sliceSag_var = tkinter.IntVar(value=200)
        self.img = np.zeros((400,400,400))
        self.img_original = np.zeros((200,200,200))
        self.img_mask = np.zeros((400,400,400))
        self.img_mask_original = np.zeros((200,200,200))
        self.img_pred_mask = np.zeros((400,400,400))
        self.img_pred_mask_original = np.zeros((200,200,200))
        self.img_result = np.zeros((400,400,400))
        self.img_result_original = np.zeros((200,200,200))
        self.sliderAxi = ''
        self.sliderCor = ''
        self.sliderSag = ''
        self.table = None
        self.last_generated = {'technique'   : 'None',
                               'architecture' : 'None',
                               'orientation': 'None',
                               }
        #self.img[20:40,20:40,20:40] = 255
        #self.img[120:140,120:140,120:140] = 255
        self.size_img = 400
        self.table_info_data = [['Attributes', 'Values']] + [['None', 'None'] for _ in range(50)]#[['Attribute','Values'],['None','None'],['None','None']]
        self.image = None
        self.type = 'flair'
        self.affine = None 
        self.shape  = None
        self.imageCor = None
        self.imageSag = None
        self.canvas = None
        self.canvasCor = None
        self.canvasSag = None
        self.canvas_image = None
        self.canvas_imageCor = None
        self.canvas_imageSag = None
        self.fmeasure_var = tk.StringVar()
        self.precision_var = tk.StringVar()
        self.recall_var = tk.StringVar()
        self.specificity_var = tk.StringVar()
        self.wmh_count_var = tk.StringVar()
        self.iou_var = tk.StringVar()
        self.hausdorff_var = tk.StringVar()
        self.dice_var = tk.StringVar()
        self.accuracy_var = tk.StringVar()
        self.wmh_load_var = tk.StringVar()
        ## Left side bar
        set_appearance_mode("light")

    def process_export_thread(self, loading_popup,pdf_exporter):
        # Call the actual process_image function in the thread
        pdf_exporter.generate_pdf()
    
        # Update the GUI after processing is complete
        self.app.after(25, lambda: self.close_loading_popup(loading_popup))

    def export_pdf(self):
        file_path_out = None
        #while file_path_out == None:
        file_path_out = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save As PDF"
        )
        parameters_dict = self.last_generated #{"param1": 0.5, "param2": 10}
        
        #print(self.img_original.shape, file_path_out, parameters_dict)
        pdf_exporter = PDFExporter(self.img, self.img_result, self.img_original.shape, file_path_out, parameters_dict)


        loading_popup = tk.Toplevel(self.app)
        loading_popup.iconbitmap('softwareLogo.ico')

        loading_popup.title("Loading")

        # Add a Label to display the loading GIF
        loading_label = tk.Label(loading_popup, text="Loading")
        loading_label.pack(pady=10)
        self.center_window(loading_popup)
        # Add your loading GIF here
        loading_gif_path = "loading.gif"
        loading_gif_frames = []
        loading_gif = Image.open(loading_gif_path)

        # Split the GIF into frames
        try:
            while True:
                loading_gif_frames.append(ImageTk.PhotoImage(loading_gif.copy()))
                loading_gif.seek(loading_gif.tell() + 1)
        except EOFError:
            pass

        # Display the animated GIF
        loading_gif_label = tk.Label(loading_popup, image=loading_gif_frames[0])
        loading_gif_label.image = loading_gif_frames[0]
        loading_gif_label.pack()

        # Start a thread to run process_image
        threading.Thread(target=self.process_export_thread, args=(loading_popup,pdf_exporter)).start()

        # Update the GIF frames periodically
        self.app.after(100, lambda: self.update_gif_frame(loading_gif_frames, loading_gif_label, 0))        # Start a thread to run process_image

        

        
    def create_sortable_table(self, master, table_data):
        columns = table_data[0]

        self.tree = ttk.Treeview(master, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_column(c, False))
            self.tree.column(col, width=100, anchor="center")

        for row_data in table_data[1:]:
            self.tree.insert("", "end", values=row_data)

        self.tree.pack(fill="both", expand=True)

    def sort_column(self, col, reverse):
        data = [(self.tree.set(child, col), child) for child in self.tree.get_children("")]
        data.sort(reverse=reverse)
        for index, item in enumerate(data):
            self.tree.move(item[1], "", index)
        self.tree.heading(col, command=lambda: self.sort_column(col, not reverse))

    def show_popup(self,event,text):

        self.popup = tk.Toplevel()
        self.popup.geometry("+{}+{}".format(event.x_root + 10, event.y_root + 10))
        self.popup.wm_overrideredirect(True)
        label = tk.Label(self.popup, text=text)
        label.pack()
    
    def hide_popup(self,event):
        
        if self.popup:
            self.popup.destroy()        
            
    def process_image(self):
        #global img
        #global file_path
        if self.type == 'flair':
            if self.file_path.endswith('nii') or self.file_path.endswith('gz'):
                self.img = nib.load(self.file_path)
                self.affine = self.img.affine
                img_original = self.img.get_fdata()
                self.img_original = img_original
            else: #dicom
                img_original = self.img_original
                self.affine = None
            self.shape = img_original.shape
            
            img_original = (img_original - img_original.min()) / (img_original.max() - img_original.min()) * 500
            
            #img_original = np.rot90(img_original, k=1, axes=(0, 1))

            # Rotate 90 degrees in the y direction
            #img_original = np.rot90(img_original, k=1, axes=(1, 2))
            
            if self.file_path.endswith('nii') or self.file_path.endswith('gz'):
                img_original = np.rot90(img_original, k=1, axes=(0, 2))
            #img_original = rotate(img_original, angle=-90, axes=(0, 1), reshape=False)
            #img_original = rotate(img_original, angle=-180, reshape=False)

            target_shape = (self.size_img, self.size_img, self.size_img)
            zoom_factors = (
                target_shape[0] / img_original.shape[0],
                target_shape[1] / img_original.shape[1],
                target_shape[2] / img_original.shape[2]
            )
            self.img = zoom(img_original, zoom_factors, order=1, mode='nearest') 
            self.img = np.stack([self.img, self.img, self.img, np.ones_like(self.img) * 255], axis=-1).astype(np.uint8)
            #this shows in the screen
            print(self.img.shape)
            self.img_result = self.img.copy()
            self.img_result_original = self.img_original.copy()
        else:
            if self.file_path.endswith('nii') or self.file_path.endswith('gz'):
                self.img_mask = nib.load(self.file_path)
                img_original = self.img_mask.get_fdata()
                self.img_mask_original = img_original
                img_original = np.rot90(img_original, k=1, axes=(0, 2))
            else:
                img_original = self.img_mask_original
            target_shape = (self.size_img, self.size_img, self.size_img)
            zoom_factors = (
                target_shape[0] / img_original.shape[0],
                target_shape[1] / img_original.shape[1],
                target_shape[2] / img_original.shape[2]
            )
            self.img_mask = zoom(img_original, zoom_factors, order=1, mode='nearest')
            self.img_mask[self.img_mask>0.5] = 1
            
            
            color_dict = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 255, 255)}
            
            colors = [color_dict.get(value, (0, 0, 0)) for value in [1, 2, 3]]
            print(type(self.img_mask[0,0,0]))
            masks = [self.img_mask == value for value in [1, 2, 3]]
            print(np.unique(masks))
            
            # Apply colors to each mask
            self.img_result = self.img.copy()
            for idx, color in enumerate(colors, start=1):
                self.img_result[masks[idx - 1], :3] = color            
            
            
            #self.img_result = self.img - self.img_mask
            #self.img_result[self.img < 0] = 0
            
        self.update_image()
        self.update_imageCor()
        self.update_imageSag()
    
        # close_loading_popup(loading_popup)
        
    def add_label(self,icon_name,info_array,text,master):
        gridtitleLong = CTkFrame(master=master, fg_color="transparent")
        gridtitleLong.pack(fill="both", padx=10, pady=(0,0))
        gridtitleLong.columnconfigure(0, weight=4)
        gridtitleLong.columnconfigure(1, weight=0)
        
        email_icon_data = Image.open(icon_name)
        email_icon = CTkImage(dark_image=email_icon_data, light_image=email_icon_data, size=(20,20))
        CTkLabel(master=gridtitleLong, text=text,image=email_icon, anchor="w", justify="left", font=("Arial", 12), text_color="#ffffff", fg_color="transparent", compound="left").grid(row=0, column=0, sticky=tk.W+tk.E, pady=(15,0))
        
        analytics_img_data = Image.open(self.path_dir+"infow.png")
        analytics_img = CTkImage(dark_image=analytics_img_data, light_image=analytics_img_data)
        CTkButton(master=gridtitleLong, image=analytics_img, text="", font=("Arial", 12), fg_color="transparent",width=1,command=lambda: self.show_info(info_array)).grid(row=0, column=1, sticky=tk.W+tk.E, pady=(15,0))
    
    def add_vert_slider(self,text,master,from_=0,to_=1,step=20):
        gridtitleLong = CTkFrame(master=master, fg_color="transparent")
        gridtitleLong.pack(fill="both", padx=0, pady=(0,0))
        gridtitleLong.columnconfigure(0, weight=4)
        gridtitleLong.columnconfigure(1, weight=0)
        
        CTkLabel(master=gridtitleLong, text=text, anchor="w", justify="left", font=("Arial", 10), text_color="#ffffff", fg_color="transparent", compound="left").grid(row=0, column=0, sticky=tk.W+tk.E, pady=(15,0))
        self.sliderThresh = CTkSlider(master=gridtitleLong,variable=self.threshold_var,from_=from_,to=to_,number_of_steps=step,
                                 button_color="#C1C3FE",progress_color="#bbbbbb")
        self.sliderThresh.grid(row=0, column=1, sticky=tk.W+tk.E, pady=(15,0))
        self.sliderThresh.bind("<Enter>", lambda event: self.show_popup(event,np.round(self.threshold_var.get(),2) ))
        self.sliderThresh.bind("<Leave>", self.hide_popup)
    
    def read_dicom_series(self):
        # Read the given DICOM slice
        first_slice = pydicom.dcmread(self.file_path)
    
        #Get attributes
        attributes = [
            [element.keyword, str(getattr(first_slice, element.keyword, ""))[:50]]
            for element in first_slice.iterall()
        ]
        
        attributes = [attr for attr in attributes if attr[1].strip() != ""]    
        
        # Get the series and study instance UIDs
        series_uid = first_slice.SeriesInstanceUID
        study_uid = first_slice.StudyInstanceUID
        
        # Get the directory containing the DICOM files
        dicom_dir = os.path.dirname(self.file_path)
        
        # Find all DICOM files in the same directory with matching series and study instance UIDs
        dicom_files = [
            os.path.join(dicom_dir, file) 
            for file in os.listdir(dicom_dir) 
            if file.endswith(".dcm") and pydicom.dcmread(os.path.join(dicom_dir, file)).SeriesInstanceUID == series_uid
        ]
        
        # Sort the files based on their instance number
        dicom_files.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)
        
        # Read each DICOM file and store the pixel data in a list
        dicom_slices = [pydicom.dcmread(file) for file in dicom_files]
        
        # Get image dimensions
        rows = int(first_slice.Rows)
        cols = int(first_slice.Columns)
        num_slices = len(dicom_slices)
        
        # Initialize 3D volume
        volume = np.zeros((num_slices, rows, cols), dtype=np.uint16)
        
        # Read pixel data from each DICOM slice and store it in the volume
        for i, dicom_slice in enumerate(dicom_slices):
            volume[i, :, :] = dicom_slice.pixel_array
        
        return [volume,attributes]

    def read_json(self):
        json_file_path = self.file_path.replace('.nii.gz','.json').replace('.nii','.json')
        print(self.file_path)
        print(json_file_path)
        
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)
        
        var = [['Attributes','Values']] + [[key, str(value)] for key, value in json_data.items()]
        print(len(var))
        #print(var)
        #oi = var.ravel()
        #for oo in oi:
        #    print(type(oo))
        
        self.table_info_data = var#[['teste','test'],['oi','oi'],['oooo','22222']]#np.array(list(json_data.items()))
        #self.table_info_data = np.array(map(str, self.table_info_data))
        #print(self.table_info_data)
        #print(self.table_info_data)
    def add_combobox(self,variable,values,master):
        grid3 = CTkFrame(master=master, fg_color="transparent")
        grid3.pack(fill="both", padx=10, pady=(0,0))
        CTkComboBox(master=grid3, variable=variable,width=230, values=values, button_color="#474BB5", border_color="#474BB5", border_width=2, button_hover_color="#AAADF0",dropdown_hover_color="#AAADF0" , dropdown_fg_color="#474BB5", dropdown_text_color="#fff").grid(row=10, column=2,  sticky="w", pady=(0,0))
    
    def show_info(self,info_array):
        info_text = "\n".join(info_array)
        messagebox.showinfo("Information", info_text)
    
    #def process_image():
    #    # Simulating a time-consuming task
    #    time.sleep(5)
    #    print("Image processing complete")
    
    def center_window(self,win):
        win.wait_visibility() # make sure the window is ready
        x = (win.winfo_screenwidth() - win.winfo_width()) // 2
        y = (win.winfo_screenheight() - win.winfo_height()) // 2
        win.geometry(f'+{x}+{y}')
        
    def load_nifti(self,typed='flair'):#flair,mask
        #global file_path
        #global img
        self.type = typed
        self.file_path = filedialog.askopenfilename()
    
        if self.file_path:
            loading_popup = tk.Toplevel(self.app)
                        
            if self.type == 'flair':
                if self.file_path.endswith('nii') or self.file_path.endswith('gz'):
                    try:
                        self.read_json()
                    except:                    
                        self.table_info_data = [['Attributes','Values'],['None','None']]
                else:
                    [volume,attributes] = self.read_dicom_series()
                    #file = pydicom.dcmread(self.file_path).pixel_array
                    self.img_original = self.img = volume
                    self.table_info_data = [['Attributes','Values']] + attributes
            else:
                if self.file_path.endswith('dcm'):
                    [volume,_] = self.read_dicom_series()
                    self.img_mask_original = volume
                #print(file)
                    
            self.update_table_data()
            loading_popup.iconbitmap('softwareLogo.ico')
    
            loading_popup.title("Loading")
    
            # Add a Label to display the loading GIF
            loading_label = tk.Label(loading_popup, text="Loading")
            loading_label.pack(pady=10)
            self.center_window(loading_popup)
            # Add your loading GIF here
            loading_gif_path = "loading.gif"
            loading_gif_frames = []
            loading_gif = Image.open(loading_gif_path)
    
            # Split the GIF into frames
            try:
                while True:
                    loading_gif_frames.append(ImageTk.PhotoImage(loading_gif.copy()))
                    loading_gif.seek(loading_gif.tell() + 1)
            except EOFError:
                pass
    
            # Display the animated GIF
            loading_gif_label = tk.Label(loading_popup, image=loading_gif_frames[0])
            loading_gif_label.image = loading_gif_frames[0]
            loading_gif_label.pack()
    
            # Start a thread to run process_image
            threading.Thread(target=self.process_image_thread, args=(loading_popup,)).start()
    
            # Update the GIF frames periodically
            self.app.after(100, lambda: self.update_gif_frame(loading_gif_frames, loading_gif_label, 0))

    def process_image_thread(self, loading_popup):
        # Call the actual process_image function in the thread
        self.process_image()
    
        # Update the GUI after processing is complete
        self.app.after(25, lambda: self.close_loading_popup(loading_popup))
    
    def update_gif_frame(self,frames, label, index):
        # Update the GIF frame
        label.configure(image=frames[index])
        label.image = frames[index]
    
        # Schedule the next update
        self.app.after(25, lambda: self.update_gif_frame(frames, label, (index + 1) % len(frames)))
    
    def close_loading_popup(self,loading_popup):
        loading_popup.destroy()  # Close the loading popu    
           
            
    def get_values(self):
        status_folder = ['2DAxi','2DCor','2DSag','2_5D']
        technique_folder = {"Traditional U-Net":"unet", "Attention U-Net":'attunet', "U-Net ++":"unet2p", "U-Net 3+":"unet3p", "LinkNet":"linknet", "FPN":"fpn", "Progressive Learning":"prlearning","Transformers":"transformers"}
        architecture_folder = {"VGG16":'VGG16', "VGG19":'VGG19', "ResNet 152":'ResNet152', "EfficientNet B0":'EfficientNetB0'}
        cross_lon_folder = {"Cross-sectional":'cross',"Longitudinal":"long"}
        return "/".join([technique_folder[self.technique_var.get()],
                        architecture_folder[self.architecture_var.get()],
                        status_folder[self.status_var.get()]])
       
        #return [st]
    def set_results(self,f_measure, iou, 
                    accuracy, recall, precision, 
                    specificity, dice, hausdorff_dist, 
                    wmh_count, wmh_size_per_region):
        self.fmeasure_var.set("{:.2f}".format(f_measure))
        self.last_generated['F-measure'] = "{:.2f}".format(f_measure)
        self.precision_var.set("{:.2f}".format(precision))   
        self.last_generated['Precision'] = "{:.2f}".format(precision)
        self.recall_var.set("{:.2f}".format(recall))      
        self.last_generated['Recall'] = "{:.2f}".format(recall)
        self.specificity_var.set("{:.2f}".format(specificity)) 
        self.last_generated['Specificity'] = "{:.2f}".format(specificity)
        self.wmh_count_var.set("{:.2f}".format(wmh_count))   
        self.last_generated['WMH count'] = "{:.2f}".format(wmh_count)
        self.iou_var.set("{:.2f}".format(iou))       
        self.last_generated['IoU'] = "{:.2f}".format(iou)
        self.hausdorff_var.set("{:.2f}".format(hausdorff_dist)) 
        self.last_generated['Hausdorff'] = "{:.2f}".format(hausdorff_dist)
        self.dice_var.set("{:.2f}".format(dice))      
        self.last_generated['Dice'] = "{:.2f}".format(dice)
        self.accuracy_var.set("{:.2f}".format(accuracy))  
        self.last_generated['Accuracy'] = "{:.2f}".format(accuracy)
        self.wmh_load_var.set("{:.2f}".format(len(wmh_size_per_region)))  
        self.last_generated['WMH Regions number'] = "{:.2f}".format(len(wmh_size_per_region))
        
    def process_seg_thread(self, loading_popup=None,withThread=True):
        # Call the actual process_image function in the thread
        
        self.generate_segmentation()
    
        # Update the GUI after processing is complete
        if withThread:
            self.app.after(25, lambda: self.close_loading_popup(loading_popup))

    def generate_segmentation_super(self):
        self.path_out = filedialog.asksaveasfilename(
            defaultextension=".nii",
            filetypes=[("Nifti format", "*.nii")],
            title="Save As Nifti"
        )
        withThread = False
        if withThread:
            loading_popup = tk.Toplevel(self.app)
            loading_popup.iconbitmap('softwareLogo.ico')
    
            loading_popup.title("Loading")
    
            # Add a Label to display the loading GIF
            loading_label = tk.Label(loading_popup, text="Loading")
            loading_label.pack(pady=10)
            self.center_window(loading_popup)
            # Add your loading GIF here
            loading_gif_path = "loading.gif"
            loading_gif_frames = []
            loading_gif = Image.open(loading_gif_path)
    
            # Split the GIF into frames
            try:
                while True:
                    loading_gif_frames.append(ImageTk.PhotoImage(loading_gif.copy()))
                    loading_gif.seek(loading_gif.tell() + 1)
            except EOFError:
                pass
    
            # Display the animated GIF
            loading_gif_label = tk.Label(loading_popup, image=loading_gif_frames[0])
            loading_gif_label.image = loading_gif_frames[0]
            loading_gif_label.pack()
        else:
            loading_popup = None 

        
        # Start a thread to run process_image
        threading.Thread(target=self.process_seg_thread, args=(loading_popup,withThread)).start()
        
        if withThread:
            # Update the GIF frames periodically
            self.app.after(100, lambda: self.update_gif_frame(loading_gif_frames, loading_gif_label, 0))        # Start a thread to run process_image

        
    def generate_segmentation(self):
        path_model = self.get_values()
        print(path_model)
        print(self.path_out)
        model = Models(path_model,path_out=self.path_out,raw_volume=self.img_original,affine=self.affine,shape=self.shape)
        self.img_pred_mask_original = model.run_pipeline()
        store = self.get_values().split('/')
        self.last_generated['technique']    =  self.technique_var.get()
        self.last_generated['architecture'] = self.architecture_var.get()
        self.last_generated['orientation'] = store[2]
        
        # organize for the screen 
        if self.file_path.endswith('nii') or self.file_path.endswith('gz'):
            self.img_pred_mask = np.rot90(self.img_pred_mask_original, k=1, axes=(0, 2))
            self.img_mask = np.rot90(self.img_mask_original, k=1, axes=(0, 2))
        else:
            self.img_pred_mask = self.img_pred_mask_original
            self.img_mask = self.img_mask_original
        np.save('FLAIR_mask_gt_oi.npy',self.img_mask)
        np.save('FLAIR_mask_pred_oi.npy',self.img_pred_mask)
        
        self.img_pred_mask = ((self.img_pred_mask > 0.5)*1).astype('uint8') #self.sliderThresh.get())
        
        if np.sum(self.img_mask) != 0:
            self.img_mask[self.img_mask>= 0.5] = 1
            self.img_mask[self.img_mask< 0.5] = 0
            self.img_mask = self.img_mask.astype('uint8')
            print(np.unique(self.img_pred_mask))
            print(np.unique(self.img_mask))
            #self.img_pred_mask = 2*self.img_pred_mask
            #print(np.unique(self.img_pred_mask))
            self.img_pred_mask = self.img_pred_mask*2 + self.img_mask
        #print(type(self.img_pred_mask[0,0,0]))
        #print(np.unique(self.img_pred_mask))
        
        target_shape = (self.size_img, self.size_img, self.size_img)
        zoom_factors = (
            target_shape[0] / self.img_pred_mask.shape[0],
            target_shape[1] / self.img_pred_mask.shape[1],
            target_shape[2] / self.img_pred_mask.shape[2]
        )
        self.img_pred_mask = zoom(self.img_pred_mask, zoom_factors, order=1, mode='nearest')
        
        color_dict = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 255, 255)}
         
        colors = [color_dict.get(value, (0, 0, 0)) for value in [1, 2, 3]]
        
        print(np.unique(self.img_pred_mask))
        masks = [self.img_pred_mask == value for value in [1, 2, 3]]
        print(np.unique(masks))
        # Apply colors to each mask
        self.img_result = self.img.copy()
        for idx, color in enumerate(colors, start=1):
            self.img_result[masks[idx - 1], :3] = color            
        
        self.update_image()
        self.update_imageCor()
        self.update_imageSag()
        
        
        if np.sum(self.img_mask) != 0:
            metrics_calculator = Metrics(self.img_pred_mask_original >0.5, self.img_mask_original>0.5)
            f_measure, iou, accuracy, recall, precision, specificity, dice, hausdorff_dist, wmh_count, wmh_size_per_region = \
                metrics_calculator.calculate_metrics()
            self.set_results(f_measure, iou, accuracy, recall, precision, specificity, dice, hausdorff_dist, wmh_count, wmh_size_per_region)
            print(f_measure, iou, accuracy, recall, precision, specificity, dice, hausdorff_dist, wmh_count, wmh_size_per_region)

    def generate_multiple_segmentation(self):
        path_model = self.get_values()
        model = Models(path_model,path_out=self.path_out,path_in=self.path_in)
        model.run_multiple_pipeline()
        
    def update_image(self,*args):
        current_slice = self.sliceAxi_var.get()
        new_photo =  ImageTk.PhotoImage(image=Image.fromarray(self.img_result[:,:,current_slice]))
        self.canvas.itemconfig(self.canvas_image, image=new_photo)
        self.canvas.photo = new_photo 
        
    def update_imageCor(self,*args):
        current_slice = self.sliceCor_var.get()
        new_photoCor =  ImageTk.PhotoImage(image=Image.fromarray(self.img_result[:,current_slice,:]))
        self.canvasCor.itemconfig(self.canvas_imageCor, image=new_photoCor)
        self.canvasCor.photo = new_photoCor 
        
    def update_imageSag(self,*args):
        current_sliceSag = self.sliceSag_var.get()
        new_photoSag = ImageTk.PhotoImage(image=Image.fromarray(self.img_result[current_sliceSag,:,:]))
        #print(np.sum(self.img[current_sliceSag,:,:]))
        self.canvasSag.itemconfig(self.canvas_imageSag, image=new_photoSag)
        self.canvasSag.photo = new_photoSag
        
    def get_settings(self):
        root2 = Toplevel()  # Create a new top-level window
        app = Settings(root2,self.path_dir)
        
        [a,b] = [app.input_dir_var.get(),app.output_dir_var.get()]
        
    def update_table_data(self):
        self.table.update_values(self.table_info_data)
        self.table.update_values(self.table_info_data)
        self.table.update_values(self.table_info_data)
        
       

    def on_resize(self, event, main_view):
        # Update the size of the cells based on the new window size
        #cell_height = int(main_view.winfo_height() / len(main_view.grid_slaves()))
        cell_width = int(main_view.winfo_width() / 2)#len(main_view.grid_slaves()[::len(main_view.grid_slaves()) // len(main_view.grid_slaves())]))
        try:
            self.canvas.place(x=cell_width/2-200,y=5)
            self.canvasCor.place(x=cell_width/2-200,y=5)
            self.canvasSag.place(x=cell_width/2-200,y=5)
            self.sliderAxi.place(x=cell_width/100,y=49)
            self.sliderCor.place(x=cell_width/100,y=49)
            self.sliderSag.place(x=cell_width/100,y=49)
        except:
            pass
           # cell.config(width=cell_width)
            
    def generate_layout(self):
        
        sidebar_frame = CTkFrame(master=self.app, fg_color="#4C52E9",  width=250, height=500, corner_radius=0)
        sidebar_frame.pack_propagate(0)
        sidebar_frame.pack(fill="y", anchor="w", side="left")


        
        # Logo in the top
        logo_img_data = Image.open(self.path_dir+"brainLogo.png")
        logo_img = CTkImage(dark_image=logo_img_data, light_image=logo_img_data, size=(77.68, 85.42))
        
        CTkLabel(master=sidebar_frame, text="", image=logo_img).pack(pady=(20, 0), anchor="center")
        CTkLabel(master=sidebar_frame, text="Margarida\nWMH Segmentation\nToolbox",font=("Arial", 26),text_color="#ffffff").pack(pady=(0, 0), anchor="center")
        
        analytics_img_data = Image.open(self.path_dir+"load_volume.png")
        analytics_img = CTkImage(dark_image=analytics_img_data, light_image=analytics_img_data)
        CTkButton(master=sidebar_frame, image=analytics_img, text="Load FLAIR volume", fg_color="#878CFA", font=("Arial Bold", 14), hover_color="#474BB5", anchor="w",command=lambda: self.load_nifti('flair')).pack(anchor="center", ipady=5, pady=(16, 0))
        
        analytics_img_data = Image.open(self.path_dir+"load_mask.png")
        analytics_img = CTkImage(dark_image=analytics_img_data, light_image=analytics_img_data)
        CTkButton(master=sidebar_frame, image=analytics_img, text="Load FLAIR mask   ", fg_color="#878CFA", font=("Arial Bold", 14), hover_color="#474BB5", anchor="w",command=lambda: self.load_nifti('mask')).pack(anchor="center", ipady=5, pady=(16, 0))
        
        #### Choose your orientation
        self.add_label("projection.png",brain_projection_info,"    Brain Projection",sidebar_frame)
        
        grid = CTkFrame(master=sidebar_frame, fg_color="transparent")
        grid.pack(fill="both", padx=10, pady=(0,0))
        sidebar_frame.rowconfigure(1, weight=0)
        
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)
        grid.columnconfigure(3, weight=1)
        CTkRadioButton(master=grid, variable=self.status_var, value=0, text="2D\nAxi", font=("Arial", 12), text_color="#ffffff", fg_color="#ff0000", border_color="#ffffff", hover_color="#ffff00").grid(row=0, column=0,  sticky="nsew", pady=(0,0))
        CTkRadioButton(master=grid, variable=self.status_var, value=1,text="2D\nCor", font=("Arial", 12), text_color="#ffffff", fg_color="#ff0000", border_color="#ffffff", hover_color="#ffff00").grid(row=0, column=1, sticky="nsew", pady=(0,0))
        CTkRadioButton(master=grid, variable=self.status_var, value=2,text="2D\nSag", font=("Arial", 12), text_color="#ffffff", fg_color="#ff0000", border_color="#ffffff", hover_color="#ffff00").grid(row=0, column=2,  sticky="w", pady=(0,0))
        CTkRadioButton(master=grid, variable=self.status_var, value=3,text="2.5D", font=("Arial", 12), text_color="#ffffff", fg_color="#ff0000", border_color="#ffffff", hover_color="#ffff00").grid(row=0, column=3,  sticky="w", pady=(0,0))
        
        #### Choose your variation
        
        self.add_label('ai.png',style_of_ai_info,'    Style of AI',sidebar_frame)
        
        #grid2 = CTkFrame(master=sidebar_frame, fg_color="transparent")
        #grid2.pack(fill="both", padx=10, pady=(0,0))
        #CTkComboBox(master=grid2, variable=,width=230, values=, button_color="#474BB5", border_color="#474BB5", border_width=2, button_hover_color="#AAADF0",dropdown_hover_color="#AAADF0" , dropdown_fg_color="#474BB5", dropdown_text_color="#fff").grid(row=10, column=2,  sticky="w", pady=(0,0))
        self.add_combobox(self.technique_var,["Traditional U-Net", "Attention U-Net", "U-Net ++", "U-Net 3+", "LinkNet", "FPN", "Progressive Learning","Transformers"],sidebar_frame)
        #### Choose your architecture
        self.add_label("architecture.png",architecture_info,"    Architecture",sidebar_frame)
        
        self.add_combobox(self.architecture_var,["VGG16", "VGG19", "ResNet 152", "EfficientNet B0"],sidebar_frame)
        #### Choose your threshold
        self.add_vert_slider("  Threshold",sidebar_frame)
        
        #### Choose your longitudinal
        self.add_label("longitudinal.png",style_of_study_info,"    Style of Study",sidebar_frame)
        
        grid3 = CTkFrame(master=sidebar_frame, fg_color="transparent")
        grid3.pack(fill="both", padx=10, pady=(0,0))
        
        grid3.columnconfigure(0, weight=1)
        grid3.columnconfigure(1, weight=1)
        CTkRadioButton(master=grid3, variable=self.cross_lon_var, value=0, text="Cross-sectional", font=("Arial", 12), text_color="#ffffff", fg_color="#ff0000", border_color="#ffffff", hover_color="#ffff00").grid(row=0, column=0,  sticky="nsew", pady=(0,0))
        CTkRadioButton(master=grid3, variable=self.cross_lon_var, value=1,text="Longitudinal", font=("Arial", 12), text_color="#ffffff", fg_color="#ff0000", border_color="#ffffff", hover_color="#ffff00").grid(row=0, column=1, sticky="nsew", pady=(0,0))

        #self.add_label("texture.png",texture_info,"    Texture",sidebar_frame)
        #self.add_combobox(self.texture_var,["GLCM", "GLRLM", "GLSZM", "NGTDM"],sidebar_frame)
        
        
        #### Generate Mask
        grid4 = CTkFrame(master=sidebar_frame, fg_color="transparent")
        grid4.pack(fill="y", padx=10, pady=(0,0))
        
        #grid4.columnconfigure(0, weight=1)
        #grid4.columnconfigure(1, weight=100)
        analytics_img_data = Image.open(self.path_dir+"segBrain.png")
        pdf_img_data = Image.open(self.path_dir+"export_pdf.png")
        analytics_img = CTkImage(dark_image=analytics_img_data, light_image=analytics_img_data)
        pdf_img = CTkImage(dark_image=pdf_img_data, light_image=pdf_img_data)
        #CTkButton(master=grid4, image=analytics_img, text="Generate Mask", fg_color="#878CFA", font=("Arial Bold", 14), hover_color="#474BB5", anchor="w",command=self.get_values).grid(row=0, column=0,  sticky="nsew", pady=(0,0))#.pack(anchor="center", ipady=5, pady=(16, 0))
        #CTkButton(master=grid4, image=pdf_img, text="", fg_color="#878CFA", font=("Arial Bold", 14), hover_color="#474BB5", anchor="w",command=self.get_values).grid(row=0, column=1,  sticky="nsew", pady=(0,0))
        CTkButton(master=grid4, image=analytics_img, text="Generate Mask", fg_color="#878CFA", font=("Arial Bold", 14), hover_color="#474BB5", anchor="w", command=self.generate_segmentation_super).grid(row=0, column=0, sticky="nsew", pady=(15,0))
        CTkLabel(master=grid4, text='     ', anchor="w", justify="left", font=("Arial", 12), text_color="#ffffff", fg_color="transparent", compound="left").grid(row=0, column=1, sticky=tk.W+tk.E, pady=(15,0))
        btpdf = CTkButton(master=grid4, image=pdf_img, text="", fg_color="#878CFA", font=("Arial Bold", 14), hover_color="#474BB5", width=10,anchor="w", command=self.export_pdf)
        btpdf.grid(row=0, column=2, sticky="w", pady=(15,0),columnspan=3)        
        btpdf.bind("<Enter>", lambda event: self.show_popup(event, 'Export PDF'))
        btpdf.bind("<Leave>", self.hide_popup)


        multiple_data = Image.open(self.path_dir+"multiple.png")
        multiple_img = CTkImage(dark_image=multiple_data, light_image=multiple_data)
        CTkButton(master=sidebar_frame, image=multiple_img, text="Run Multiple Patients", fg_color="transparent", font=("Arial Bold", 14), command=lambda: MultPatients.show_custom_tkinter_screen(self.get_values(),self.path_dir),hover_color="#474BB5", anchor="w").pack(anchor="center", ipady=5, pady=(8, 0))
        
        #### Set paths
        settings_img_data = Image.open(self.path_dir+"settings_icon.png")
        settings_img = CTkImage(dark_image=settings_img_data, light_image=settings_img_data)
        CTkButton(master=sidebar_frame, image=settings_img, text="Settings", fg_color="transparent", font=("Arial Bold", 14), hover_color="#474BB5", anchor="w",command=self.get_settings).pack(anchor="center", ipady=5, pady=(16, 0))


                
        # person_img_data = Image.open(self.path_dir+"person_icon.png")
        # person_img = CTkImage(dark_image=person_img_data, light_image=person_img_data)
        # CTkButton(master=sidebar_frame, image=person_img, text="Account", fg_color="transparent", font=("Arial Bold", 14), hover_color="#207244", anchor="w").pack(anchor="center", ipady=5, pady=(30, 0))
        
        
        
        
        # ##################################################################### BODY
        # main_view = CTkFrame(master=self.app, fg_color="#fff",  width=1280, height=650, corner_radius=0)
        # main_view.pack_propagate(0)
        # main_view.pack(fill="both", anchor="w", side="left")
        
        
        # #Create the grid within the main_frame
        # rows = 2
        # cols = 2
        # grid_height_percent = 1
        
        # cell_height = int(main_view.winfo_reqheight() * grid_height_percent/2)
        # cell_width = int(main_view.winfo_reqwidth()/2)
        
        # for row in range(0, rows):
        #     for col in range(cols):
        #         x0 = col * cell_width
        #         y0 = row * cell_height
        #         x1 = (col + 1) * cell_width
        #         y1 = (row + 2) * cell_height
        
        #         cell = tk.Canvas(main_view, width=cell_width, height=cell_height, bg="black", highlightthickness=2, highlightbackground="white")
        #         cell.place(x=x0, y=y0)

        main_view = CTkFrame(master=self.app, width=1280, height=650)
        main_view.pack_propagate(0)
        main_view.pack(fill="both", anchor="w", side="left",expand=True)       
        rows = 2
        cols = 2
        grid_height_percent = 1
        
        cell_height = int(main_view.winfo_reqheight() * grid_height_percent / rows)
        cell_width = int(main_view.winfo_reqwidth() / cols)
        cells = []
        
        for row in range(rows):
            for col in range(cols):
                x0 = col * cell_width
                y0 = row * cell_height
                x1 = (col + 1) * cell_width
                y1 = (row + 1) * cell_height
    
                cell = tk.Canvas(main_view, width=cell_width, height=cell_height, bg="black", highlightthickness=2, highlightbackground="white")
                cells.append(cell)
                cell.grid(row=row, column=col, sticky="nsew")
                
        # Configure row and column weights to allow resizing
        #for i in range(rows):
        #    main_view.grid_rowconfigure(i, weight=1)
        for i in range(cols):
            main_view.grid_columnconfigure(i, weight=1)

        # Bind the resizing event
        self.app.bind("<Configure>", lambda event, mv=main_view: self.on_resize(event, main_view))


        # Add a table underneath the canvas
        # table = ttk.Treeview(main_view)
        # table.grid(row=3, column=0, sticky="nsew")
        # table["columns"] = ("1", "2", "3", "4")
        # table.column("0", width=0, stretch=tk.NO)
        # table.column("1", anchor=tk.W, width=100)
        # table.column("2", anchor=tk.W, width=100)
        # table.column("3", anchor=tk.W, width=100)
        # table.column("4", anchor=tk.W, width=100)
        
        # table_data = [
        #     ["Order ID", "Item Name", "Customer", "Address", "Status", "Quantity"],
        #     ['3833', 'Smartphone', 'Alice', '123 Main St', 'Confirmed', '8'],
        #     ['6432', 'Laptop', 'Bob', '456 Elm St', 'Packing', '5'],
        #     ['2180', 'Tablet', 'Crystal', '789 Oak St', 'Delivered', '1'],
        #     ['5438', 'Headphones', 'John', '101 Pine St', 'Confirmed', '9'],
        #     ['9144', 'Camera', 'David', '202 Cedar St', 'Processing', '2'],
        #     ['7689', 'Printer', 'Alice', '303 Maple St', 'Cancelled', '2'],
        #     ['1323', 'Smartwatch', 'Crystal', '404 Birch St', 'Shipping', '6'],
        #     ['7391', 'Keyboard', 'John', '505 Redwood St', 'Cancelled', '10'],
        #     ['4915', 'Monitor', 'Alice', '606 Fir St', 'Shipping', '6'],
        #     ['5548', 'External Hard Drive', 'David', '707 Oak St', 'Delivered', '10'],
        #     ['5485', 'Table Lamp', 'Crystal', '808 Pine St', 'Confirmed', '4'],
        #     ['7764', 'Desk Chair', 'Bob', '909 Cedar St', 'Processing', '9'],
        #     ['8252', 'Coffee Maker', 'John', '1010 Elm St', 'Confirmed', '6'],
        #     ['2377', 'Blender', 'David', '1111 Redwood St', 'Shipping', '2'],
        #     ['5287', 'Toaster', 'Alice', '1212 Maple St', 'Processing', '1'],
        #     ['7739', 'Microwave', 'Crystal', '1313 Cedar St', 'Confirmed', '8'],
        #     ['3129', 'Refrigerator', 'John', '1414 Oak St', 'Processing', '5'],
        #     ['4789', 'Vacuum Cleaner', 'Bob', '1515 Pine St', 'Cancelled', '10']
        # ]
        
        #working
        table_frame = CTkScrollableFrame(master=main_view, fg_color="transparent")
        table_frame.grid(row=2,column=0,sticky='nsew')
        self.table = CTkTable(master=table_frame, values=self.table_info_data, colors=["#E6E6E6", "#EEEEEE"], header_color="#4C52E9", hover_color="#B4B4B4")
        self.table.edit_row(0, text_color="#fff", hover_color="#4C52E9")
        self.table.pack(expand=True)
        
        #self.fmeasure=10
        metrics_frame = CTkScrollableFrame(master=main_view, fg_color="transparent")
        metrics_frame.grid(row=2,column=1,sticky='nsew')
        CTkLabel(master=metrics_frame,text='F-Measure').grid(row=0,column=0)
        CTkLabel(master=metrics_frame,text='Precision').grid(row=0,column=1)
        CTkLabel(master=metrics_frame,text='Recall').grid(row=0,column=2)
        CTkLabel(master=metrics_frame,text='Specificity').grid(row=0,column=3)
        CTkLabel(master=metrics_frame,text='WMH Count').grid(row=0,column=4)        
        CTkEntry(master=metrics_frame,textvariable=self.fmeasure_var,width=120).grid(row=1,column=0)
        CTkEntry(master=metrics_frame,textvariable=self.precision_var,width=120).grid(row=1,column=1)
        CTkEntry(master=metrics_frame,textvariable=self.recall_var,width=120).grid(row=1,column=2)
        CTkEntry(master=metrics_frame,textvariable=self.specificity_var,width=120).grid(row=1,column=3)
        CTkEntry(master=metrics_frame,textvariable=self.wmh_count_var,width=120).grid(row=1,column=4)
        CTkLabel(master=metrics_frame,text='IoU').grid(row=2,column=0)
        CTkLabel(master=metrics_frame,text='Hausdorff').grid(row=2,column=1)
        CTkLabel(master=metrics_frame,text='Dice').grid(row=2,column=2)        
        CTkLabel(master=metrics_frame,text='Accuracy').grid(row=2,column=3)
        CTkLabel(master=metrics_frame,text='WMH Load').grid(row=2,column=4)
        CTkEntry(master=metrics_frame,textvariable=self.iou_var,width=120).grid(row=3,column=0)
        CTkEntry(master=metrics_frame,textvariable=self.hausdorff_var,width=120).grid(row=3,column=1)
        CTkEntry(master=metrics_frame,textvariable=self.dice_var,width=120).grid(row=3,column=2)        
        CTkEntry(master=metrics_frame,textvariable=self.accuracy_var,width=120).grid(row=3,column=3)        
        CTkEntry(master=metrics_frame,textvariable=self.wmh_load_var,width=120).grid(row=3,column=4)    

        #CTkButton(master=metrics_frame, image=analytics_img, text="Generate Mask", fg_color="#878CFA", font=("Arial Bold", 14), hover_color="#474BB5", anchor="w", command=self.generate_segmentation).pack()
       ####
        

        # table_frame = CTkScrollableFrame(master=main_view, fg_color="transparent")
        # table_frame.grid(row=2, column=0, sticky='nsew')

        # # Use the create_sortable_table method
        # self.create_sortable_table(master=table_frame, table_data=table_data)
        

        size_img = 400
        #file_path = 'orig.nii'
        

        [Sag_len,Cor_len,Axi_len] = self.img.shape
        
        # sliceAxi_var = tkinter.IntVar(value=200)
        
        self.sliderAxi= CTkSlider(master=cells[0],variable=self.sliceAxi_var,from_=0,to=Axi_len-1,number_of_steps=Axi_len-1,
                                button_color="#C1C3FE",bg_color='black',progress_color="#bbbbbb",orientation='vertical')#.pack()#.grid(row=0, column=1, sticky=tk.W+tk.E, pady=(15,0))
        
        # sliceCor_var = tkinter.IntVar(value=200)
        
        self.sliderCor= CTkSlider(master=cells[1],variable=self.sliceCor_var,from_=0,to=Cor_len-1,number_of_steps=Cor_len-1,
                                button_color="#C1C3FE",bg_color='black',progress_color="#bbbbbb",orientation='vertical')#.pack()#.grid(row=0, column=1, sticky=tk.W+tk.E, pady=(15,0))

        # sliceSag_var = tkinter.IntVar(value=200)
        
        self.sliderSag= CTkSlider(master=cells[2],variable=self.sliceSag_var,from_=0,to=Sag_len-1,number_of_steps=Sag_len-1,
                                button_color="#C1C3FE",bg_color='black',progress_color="#bbbbbb",orientation='vertical')#.pack()#.grid(row=0, column=1, sticky=tk.W+tk.E, pady=(15,0))
        
        
        #print(np.sum(self.img[:,:,130]))
        self.image =  ImageTk.PhotoImage(image=Image.fromarray(self.img[:,:,130]))
        self.canvas = tk.Canvas(cells[0],width=size_img,height=size_img,bd=0, highlightthickness=0)
        self.canvas.pack()
        self.canvas_image = self.canvas.create_image(0,0, anchor="nw", image=self.image)
        self.canvas.place(x=200,y=5)
        
        self.imageCor = ImageTk.PhotoImage(image=Image.fromarray(self.img[:,130,:]))
        self.canvasCor = tk.Canvas(cells[1],width=size_img,height=size_img,bd=0, highlightthickness=0)
        self.canvasCor.pack()
        self.canvas_imageCor = self.canvasCor.create_image(0,0, anchor="nw", image=self.imageCor)
        self.canvasCor.place(x=200,y=5)
        
        self.imageSag =  ImageTk.PhotoImage(image=Image.fromarray(self.img[130,:,:]))
        self.canvasSag = tk.Canvas(cells[2],width=size_img,height=size_img,bd=0, highlightthickness=0)
        self.canvasSag.pack()
        self.canvas_imageSag = self.canvasSag.create_image(0,0, anchor="nw", image=self.imageSag)
        self.canvasSag.place(x=200,y=5)
        
        self.sliceAxi_var.trace_add("write", self.update_image)
        self.sliceCor_var.trace_add("write", self.update_imageCor)
        self.sliceSag_var.trace_add("write", self.update_imageSag)
        self.sliderAxi.place(x=610,y=49)
        self.sliderCor.place(x=1250,y=49)
        self.sliderSag.place(x=610,y=380)        
        # grid4 = CTkFrame(master=main_view, fg_color="transparent")
        # grid4.pack(fill="y", padx=10, pady=(0,0))

    # Run the application
    def run(self):
        self.generate_layout()
        self.app.mainloop()
    
if __name__ == '__main__':
    wmh_app = MargaridaWMHSeg()
    wmh_app.run()