import sys
import numpy as np
from PIL import Image
from cellpose import models

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
    
    # Define parameters for Cellpose
    diameter = 30.0  # You can adjust this based on your data
    flow_threshold = 0.4  # Typical value for flow threshold
    cellprob_threshold = 0.0  # Minimum value for cell probability

    # Apply Cellpose model
    try:
        masks, flows, styles, diams = model.eval(
            images,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )
    except TypeError as e:
        print(f"Error during model evaluation: {e}")
        sys.exit(1)

    # Save the results manually
    output_path = image_path.replace('.tif', '_cellpose.tif')
    num_slices = masks.shape[0]  # Number of slices
    for i in range(num_slices):
        mask_image = Image.fromarray(masks[i].astype(np.uint16))  # Convert mask to uint16 for TIFF
        if i == 0:
            mask_image.save(output_path, save_all=True, append_images=[Image.fromarray(masks[j].astype(np.uint16)) for j in range(1, num_slices)])
        else:
            mask_image.save(output_path, save_all=True, append_images=[Image.fromarray(masks[j].astype(np.uint16)) for j in range(i, num_slices)])
    
    print(f"Processed image saved as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]  # Get the image path from the command line arguments
    apply_model(image_path)
