import sys
import numpy as np
from PIL import Image
from stardist.models import StarDist2D
from stardist import export_imagej_rois

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

    # Apply StarDist model
    masks = []
    for i in range(images.shape[0]):
        img_slice = images[i]
        labels, img = model.predict_instances(img_slice)

        save_tiff_imagej_compatible(f'img{i}.tif', img, axes='ZYX')
        save_tiff_imagej_compatible(f'img{i}_labels.tif', labels, axes='ZYX')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_stardist.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]  # Get the image path from the command line arguments
    apply_model(image_path)
