# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:59:37 2023

@author: kaueu
"""

import nibabel as nib
import numpy as np
import glob
import sys
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import segmentation_models_3D as sm
from tensorflow.keras.models import load_model
from pdf_exporter import PDFExporter
import argparse

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, text):
        # Call the callback function with the captured output
        if self.text_widget:
            self.text_widget.insert('end',text)
            self.text_widget.see('end')
            
class Models:
    def __init__(self,path_model, path_out, raw_volume=None, affine=None, shape=None, path_in=None, path_pdf=None, text_widget=None):
        self.path_model   = path_model        
        self.path_out     = path_out
        self.path_in      = path_in
        self.path_pdf     = path_pdf
        self.filename     = ''
        self.volume       = raw_volume
        self.volume_result= None
        self.affine       = affine
        self.shape        = shape
        self.new_shape    = None
        self.crops_number = None
        self.text_widget  = text_widget
        self.attrib_model = {'technique'   : path_model.split('/')[0],
                             'architecture' : path_model.split('/')[1],
                             'orientation': path_model.split('/')[2],
                            }

        
    def printCB(self,text):
        original_stdout = sys.stdout
        sys.stdout = StdoutRedirector(self.text_widget) 
        print(text)
        sys.stdout = original_stdout          
    def print_vars(self):

        self.printCB(self.path_model,self.path_out,self.path_in,self.volume,self.affine,self.shape,self.attrib_model)


    def read_volume(self):
        self.volume = nib.load(self.path_in+'/'+self.filename)
        self.affine = self.volume.affine
        self.shape  = self.volume.shape
        self.volume = self.volume.get_fdata()

    def calculate_percentile_image(self,percentile):
        percent = np.percentile(self.volume.ravel(),percentile)
        percentile_image = self.volume / percent
        percentile_image[percentile_image>1] = 1
        return percentile_image
    
    def add_blank_slices(self):
        self.printCB('add blank slices')
        [shapeX,shapeY,shapeZ] = self.shape
        self.volume = self.calculate_percentile_image(98)
        reshaped_volume  = np.pad(self.volume, ((0,64 - (shapeX % 64) if shapeX % 64 != 0 else 0),
                                              (0,64 - (shapeY % 64) if shapeY % 64 != 0 else 0),
                                              (0,64 - (shapeZ % 64) if shapeZ % 64 != 0 else 0)),'constant')
        print(self.shape)
        print(reshaped_volume.shape)
        self.new_shape = reshaped_volume.shape
        return reshaped_volume
    
    def crop_volume(self,reshaped_volume):
        self.printCB('crop volume')
        img_patches = patchify(reshaped_volume, (64,64,64), step=64)
        self.crops_number = img_patches.shape
        cropped_volume = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
        print(cropped_volume.shape)
        return cropped_volume

    def slice_volume(self,cropped_volume):
        self.printCB('slice volume')
        if self.attrib_model['orientation'] == '2DAxi':
            sliced_volume = cropped_volume.transpose((0, 3, 1, 2)).reshape(-1, 64, 64)
        elif self.attrib_model['orientation'] == '2DCor':
            sliced_volume = cropped_volume.transpose((0, 2, 1, 3)).reshape(-1, 64, 64)
        elif self.attrib_model['orientation'] == '2DSag':
            sliced_volume = cropped_volume.transpose((0, 1, 2, 3)).reshape(-1, 64, 64)
        print(sliced_volume.shape)
        return sliced_volume
    
    def process_to_pdf_show(self):
        pass
    
    def run_model(self,sliced_volume):
        self.printCB('run model')
        dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 0.5]))
        focal_loss = sm.losses.BinaryFocalLoss()
        model = load_model('models/'+self.path_model,custom_objects={'iou_score':sm.metrics.IOUScore(),'f1-score':sm.metrics.FScore(),'dice_loss_plus_1binary_focal_loss' :(dice_loss + (1 * focal_loss))})
        sliced_seg_volume=model.predict(sliced_volume)
        #sliced_seg_volume = sliced_volume
        
        return sliced_seg_volume

    def rebuild_volume(self,sliced_seg_volume):
        self.printCB('rebuild_volume')
        #print(self.crops_number)

        number_of_crops = self.crops_number[0] * self.crops_number[1] * self.crops_number[2]
        if self.attrib_model['orientation'] == '2DAxi':                       
            cropped_seg_volume = sliced_seg_volume.reshape(number_of_crops, 64, 64, 64).transpose((0, 2, 3, 1))
        elif self.attrib_model['orientation'] == '2DCor':                                  
            cropped_seg_volume = sliced_seg_volume.reshape(number_of_crops, 64, 64, 64).transpose((0, 2, 1, 3))
        elif self.attrib_model['orientation'] == '2DSag':
            cropped_seg_volume = sliced_seg_volume.reshape(number_of_crops, 64, 64, 64).transpose((0, 1, 2, 3))

        patched_seg_volume = cropped_seg_volume.reshape(self.crops_number)
        print(patched_seg_volume.shape)       
        seg_volume_aux = unpatchify(patched_seg_volume,self.new_shape)
        print(seg_volume_aux.shape)
        seg_volume = seg_volume_aux[:self.shape[0],
                                    :self.shape[1],
                                    :self.shape[2]]
        print(seg_volume.shape)
    
            
        return seg_volume

    def save_volume(self,seg_volume): 
        self.printCB('save_volume')
        #print(self.save_volume)
        file_Nifti  = nib.Nifti1Image(seg_volume.astype('float32'), affine=self.affine)
        if self.path_out.endswith('.nii'):
            nib.save(file_Nifti,self.path_out)                
        else:
            nib.save(file_Nifti,self.path_out+'/'+self.filename)                        
    
    def export_pdf(self,seg_volume):
        self.printCB('export_pdf')
        
        self.volume = ((self.volume - self.volume.min()) / (self.volume.max() - self.volume.min())) * 100
        volume_reshaped = np.stack([self.volume, self.volume, self.volume, 
                                np.ones_like(self.volume) * 255], axis=-1).astype(np.uint8)
        
        volume_result = volume_reshaped.copy()
        seg_volume[seg_volume>=0.5] = 1
        seg_volume[seg_volume<0.5]  = 0
        masks = [seg_volume == value for value in [1, 2, 3]]
       
        color_dict = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 255, 255)}
        colors = [color_dict.get(value, (0, 0, 0)) for value in [1, 2, 3]]
        for idx, color in enumerate(colors, start=1):
            volume_result[masks[idx - 1], :3] = color         
        pdfexport = PDFExporter(volume_reshaped, volume_result, self.volume.shape, self.path_pdf+'/'+self.filename.replace('.nii','.pdf'), {'':''})
        pdfexport.generate_pdf()
        #pass

    def run_pipeline(self):
        reshaped_volume   = self.add_blank_slices()   #  
        cropped_volume    = self.crop_volume(reshaped_volume) #
        sliced_volume     = self.slice_volume(cropped_volume)
        sliced_seg_volume = self.run_model(sliced_volume)
        seg_volume        = self.rebuild_volume(sliced_seg_volume)
        self.save_volume(seg_volume)
        return seg_volume

    def run_multiple_pipeline(self):
        path_files = glob.glob(self.path_in+'/*.nii')
        for path_file in path_files:
            self.filename = path_file.replace('\\','/').split('/')[-1]
            self.printCB('Processing: '+self.filename)
            self.read_volume()
            seg_volume = self.run_pipeline()
            if self.path_pdf:
                self.export_pdf(seg_volume)
        


def main():
    parser = argparse.ArgumentParser(description="Segment WMHs using machine learning models.")
    parser.add_argument("-in", dest="path_in", required=True, help="Input folder with 3D volumes in NII.")
    parser.add_argument("-out", dest="path_out", required=True, help="Output folder for segmented 3D volumes.")
    parser.add_argument("-model", dest="model_name", required=True, choices=['unet', 'attunet', 'unet2p', 'unet3p', 'linknet', 'fpn', 'prlearning', 'transformers'], help="Choose a model.")
    parser.add_argument("-architecture", dest="architecture_name", required=True, choices=['vgg16', 'vgg19', 'resnet152', 'effnetb0'], help="Choose an architecture.")
    parser.add_argument("-orientation", dest="orientation_name", required=True, choices=['2DAxi', '2DCor', '2DSag', '2.5D'], help="Choose an orientation.")
    parser.add_argument("-pdf", dest="path_pdf", default=None, help="Optional: Folder to save a segmentation report in PDF format.")

    args = parser.parse_args()

    path_model = f"{args.model_name}/{args.architecture_name}/{args.orientation_name}"
    path_out = args.path_out
    path_in = args.path_in
    path_pdf = args.path_pdf

    print(f"Running segmentation with the following parameters:")
    print(f"  path_model: {path_model}")
    print(f"  path_out: {path_out}")
    print(f"  path_in: {path_in}")
    print(f"  path_pdf: {path_pdf}")
    
    model = Models(path_model=path_model, 
                   path_out=path_out,
                   path_in=path_in,
                   path_pdf=path_pdf)
    model.run_multiple_pipeline()

if __name__ == "__main__":
    main()   
        
