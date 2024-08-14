import sys
import numpy as np
from PIL import Image
from stardist.models import StarDist2D
from skimage import io

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

    # Initialize StarDist model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    
    # Define parameters for StarDist
    prob_thresh = 0.5  # Adjust this threshold based on your data

    # Apply StarDist model
    masks = []
    for i in range(images.shape[0]):
        img_slice = images[i]
        labels, _ = model.predict(img_slice, prob_thresh=prob_thresh)
        masks.append(labels)
    
    masks = np.stack(masks, axis=0)

    # Save the results manually
    output_path = image_path.replace('.tif', '_stardist.tif')
    for i in range(masks.shape[0]):
        slice_output_path = f"{output_path}_{i}.tif"
        io.imsave(slice_output_path, masks[i].astype(np.uint16))
  
    print(f"Processed images saved as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_stardist.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]  # Get the image path from the command line arguments
    apply_model(image_path)
