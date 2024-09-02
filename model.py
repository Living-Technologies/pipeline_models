import cv2
import numpy as np
import sys
from PIL import Image

def apply_model(image_path):
    # Open the TIFF file using PIL to handle multi-page TIFFs
    img = Image.open(image_path)
    
    # Initialize an array to store the blurred images
    blurred_images = []

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
        
        # Apply Gaussian blur
        blurred_img = cv2.GaussianBlur(img_slice, (5, 5), 0)
        
        # Append the blurred slice to the list
        blurred_images.append(Image.fromarray(blurred_img))

    # Save the blurred images as a new multi-page TIFF
    output_path = image_path.replace('temp', 'gaussian')
    blurred_images[0].save(output_path, save_all=True, append_images=blurred_images[1:])
    print(f"Gaussian-blurred image saved as {output_path}")

if __name__ == "__main__":
    image_path = sys.argv[1]  # Get the image path from the command line arguments
    apply_model(image_path)
