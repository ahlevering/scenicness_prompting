import os
from PIL import Image

data_path = "data/maps/"

# Get a list of all PNG files in the directory
image_files = [f for f in os.listdir(data_path) if f.endswith(".png")]

# Iterate over each image file
for filename in image_files:
    # Load the image
    image = Image.open(os.path.join(data_path, filename))

    # Calculate crop dimensions
    width, height = image.size
    left_crop = int(width * 0.2)  # Crop 20% from the left side
    right_crop = int(width * 0.3)  # Crop 30% from the right side

    # Crop the image
    cropped_image = image.crop((left_crop, 0, width - right_crop, height))

    # Save the cropped image with the "_cropped" suffix
    cropped_filename = os.path.splitext(filename)[0] + "_cropped.png"
    cropped_image.save(os.path.join(data_path, cropped_filename), dpi=(100, 100))

    print(f"Image '{filename}' cropped and saved as '{cropped_filename}'")