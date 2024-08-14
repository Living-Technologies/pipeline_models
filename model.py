import sys
import numpy as np
from PIL import Image
from cellpose import models
from skimage import io
import cv2

def apply_model(image_path):
    # Open the TIFF file using PIL to handle multi-page TIFFs
    img = Image.open(image_path)
    
    # Initialize an array to store the segmented images
    segmented_images = []

    # Loop through each slice (page) in the TIFF
    for i in range(img.n_frames):
        img.seek(i)  # Move to the ith slice
        img_slice = np.array(img)  # Convert the slice to a NumPy array
        
        # Convert to 8-bit single channel if necessary
        if img_slice.dtype != np.uint8:
            # Normalize the image to 0-255 range, then convert to uint8
            img_slice = cv2.normalize(img_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # If the image has multiple channels (e.g., RGB), convert to grayscale
        if len(img_slice.shape) > 2:
            img_slice = cv2.cvtColor(img_slice, cv2.COLOR_BGR2GRAY)
        
        # Initialize the Cellpose model (use 'cyto' or 'nuclei' based on your data)
        model = models.Cellpose(gpu=False, model_type='cyto')
        
        # Segment the image using Cellpose
        masks, flows, styles, diams = model.eval(img_slice, diameter=15.0, flow_threshold=0.4, cellprob_threshold=0.0)
        
        # Append the segmented mask to the list
        segmented_images.append(Image.fromarray(masks.astype(np.uint16)))

    # Save the segmented masks as a new multi-page TIFF
    output_path = image_path.replace('.tif', '_cellpose.tif')
    segmented_images[0].save(output_path, save_all=True, append_images=segmented_images[1:])
    print(f"Segmented image saved as {output_path}")

if __name__ == "__main__":
    image_path = sys.argv[1]  # Get the image path from the command line arguments
    apply_model(image_path)
