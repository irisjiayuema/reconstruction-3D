import cv2
import os

def convert_images(directory):
    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            # Construct full path to the file
            img_path = os.path.join(directory, filename)
            # Open the image
            img = cv2.imread(img_path, -1)
            # Check if it's a valid image
            if img is not None:
                # Convert RGBA to RGB
                if img.shape[-1] == 4:  # Check if the image has an alpha channel
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                # Prepare the output path
                out_path = img_path.replace(".png", ".raw")
                # Open the output file
                with open(out_path, 'wb') as out:
                    # Write the image data to a .raw file
                    img.tofile(out)
                print(f"Converted {img_path} to {out_path}")
            else:
                print(f"Failed to load image {img_path}")

# Directories containing images
directories = [
    
    "/home/rsp8/scratch/data/u4k/00006/Image0",
    "/home/rsp8/scratch/data/u4k/00006/Image1"
]

# Process each directory
for dir_path in directories:
    convert_images(dir_path)
