import numpy as np
from fpdf import FPDF
from io import BytesIO
import os
import matplotlib.pyplot as plt
import nibabel as nib
class PDFExporter:
    def __init__(self, volume_in, volume_mask, shape, path_out, parameters_dict):
        self.volume_in = volume_in
        self.volume_mask = volume_mask
        self.shape = shape
        self.path_out = path_out
        self.parameters_dict = parameters_dict

    def generate_pdf(self):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # First page content
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Processed: {self.path_out}", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Volume In shape: {self.shape}", ln=True, align='C')
        
        parameters_str = f"Parameters: {self.parameters_dict}".replace('{','').replace("'","").replace('}','')
        #print(parameters_str)
        chunk_size = 90
        chunks = [parameters_str[i:i+chunk_size] for i in range(0, len(parameters_str), chunk_size)]

        for chunk in chunks:
            pdf.cell(200, 10, txt=chunk, ln=True, align='C')

        # Grid plot for volume_in
        self.plot_grid(pdf, 0, 2, self.volume_in)

        # Grid plot for volume_mask
        self.plot_grid(pdf, 3, 5, self.volume_mask)

        pdf_filename = self.path_out
        pdf.output(pdf_filename)
        print(f"PDF generated successfully: {pdf_filename}")

    def plot_grid(self, pdf, start_col, end_col, volume):
        for col in range(start_col, end_col + 1):
            max_index = volume.shape[col - start_col]

            for row in range(10):
                index = int(row * max_index / 10)

                # Set position and image on PDF
                pdf.set_xy(col * 30+15, (row * 20)+80)
                #pdf.cell(60, 40, txt=f"Slice {index}\nalong axis {col}"[:30], ln=True)

                # Get the image as a BytesIO object
                image_bytes = self.get_image_bytes(volume, col, index)

                # Save BytesIO to a temporary file
                temp_filename = f"temp_slice_{col}_{index}.png"
                with open(temp_filename, 'wb') as temp_file:
                    temp_file.write(image_bytes.getvalue())

                # Use the temporary file in pdf.image()
                pdf.image(temp_filename, x=pdf.get_x(), y=pdf.get_y(), w=30, h=20)

                # Remove the temporary file
                os.remove(temp_filename)

    def get_image_bytes(self, volume, axis, index):
        # Wrap around the index if it exceeds the maximum value
        index = index % volume.shape[axis%3]

        # Create a BytesIO object to store the image
        image_buffer = BytesIO()

        # Save the slice as an in-memory PNG
        plt.imshow(volume.take(index, axis=axis%3))#,cmap='Greys_r')
        plt.axis('off')
        plt.savefig(image_buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # Reset the buffer position to the beginning
        image_buffer.seek(0)

        return image_buffer

if __name__ == "__main__":
    # Example usage
    volume_in = nib.load('../FLAIR_std.nii').get_fdata()#np.random.rand(10, 20, 15)
    volume_mask =  nib.load('../FLAIR_mask_std.nii').get_fdata()#np.random.rand(10, 20, 15)
    path_out = "example_patient.pdf"
    parameters_dict = {"param1": 0.5, "param2": 10,"param3": 0.5, "param4": 10,"param5": 0.5, "param6": 10,"param7": 0.5, "param8": 10,
                       "param9": 0.5, "param22": 10,"param12": 0.5, "param22": 10,"param17": 0.5, "param83": 10,"param145": 0.5, "param2234": 10}

    pdf_exporter = PDFExporter(volume_in, volume_mask, volume_in.shape, path_out, parameters_dict)
    pdf_exporter.generate_pdf()
