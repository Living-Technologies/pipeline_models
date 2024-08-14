import sys
import os
import numpy as np
import cv2
from PIL import Image
from cellpose import models, io

def apply_model(image_path):
    # Load the image stack
    img = Image.open(image_path)
    
    # Initialize an array to store the images
    images = []
    
    # Loop through each slice (page) in the TIFF stack
    for i in range(img.n_frames):
        img.seek(i)  # Move to the ith slice
        img_slice = np.array(img)  # Convert the slice to a NumPy array
        images.append(img_slice)
    
    # Convert list of images to a NumPy array
    images = np.stack(images, axis=0)

    # Initialize Cellpose model
    model = models.Cellpose(gpu=False, model_type='cyto')
    
    # Apply Cellpose model
    masks, flows, styles, diams = model.eval(images, diameter=None, flow_threshold=None, cellprob_threshold=None)

    # Save the results as a new multi-page TIFF
    output_path = image_path.replace('.tif', '_cellpose.tif')
    io.masks_flows_to_tiff(output_path, masks, flows, images)

    print(f"Processed image saved as {output_path}")

if __name__ == "__main__":
    image_path = sys.argv[1]  # Get the image path from the command line arguments
    apply_model(image_path)
