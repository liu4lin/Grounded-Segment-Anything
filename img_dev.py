import PIL
from PIL import Image, ImageFilter
import os

def copy_paste_with_smooth_filter(background_image, foreground_image, mask_image, position):
    # Create a new image to paste the foreground onto
    new_image = Image.new('RGBA', background_image.size, (0, 0, 0, 0))

    # Paste the foreground image onto the new image based on the mask
    new_image.paste(foreground_image, position, mask_image)

    # Iterate through each pixel of the new image and apply a smooth filter to a window of pixels
    # that have a mixture mask value with the boundary mask
    for x in range(new_image.width):
        for y in range(new_image.height):
            # Get the mask value at the current pixel
            mask_value = mask_image.getpixel((x, y))

            # If the mask value is in the mixture range, check the window for a mixture of True and False mask values
            if mask_value > 0 and mask_value < 255:
                # Define the window size for the smooth filter
                window_size = 3

                # Get the pixel values for the current window
                window_box = (x-window_size, y-window_size, x+window_size, y+window_size)
                window_mask_region = mask_image.crop(window_box)

                # Check if the window mask region has a mixture of True and False mask values
                if not all(pixel == True or pixel == False for pixel in window_mask_region.getdata()):
                    # Apply the smooth filter to the window region
                    window_region = new_image.crop(window_box)
                    smoothed_window_region = window_region.filter(ImageFilter.SMOOTH)
                    new_image.paste(smoothed_window_region, window_box)

    # Convert the background image to "RGBA" mode if it is not already
    if background_image.mode != 'RGBA':
        background_image = background_image.convert('RGBA')

    # Convert the new image to "RGBA" mode if it is not already
    if new_image.mode != 'RGBA':
        new_image = new_image.convert('RGBA')

    # Paste the new image onto the background image
    result_image = Image.alpha_composite(background_image, new_image)
    result_image = result_image.convert('RGB')
    return result_image





#image_pil.paste(image, (0, 0), mask_pil)
output_dir = 'outputs'
image_pil = Image.open('outputs/raw_image.jpg')
image = Image.open('outputs/inpainting.jpg')
mask_pil = Image.open('outputs/mask.jpg')
#image = image.resize(image_pil.size)
#image_pil = copy_paste_with_smooth_filter(image_pil, image, mask_pil, (0, 0))
#image_pil.save(os.path.join(output_dir, "debug.jpg"))

#mask_pil = PIL.ImageOps.invert(mask_pil)


import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening

# Load the binary mask image
mask_image = Image.open('outputs/tmpdet1afba.png').convert('1')

# Convert the mask image to a numpy array
mask_array = np.array(mask_image, dtype=np.uint8)

# Perform an opening operation to remove sparse noise regions
opened_mask_array = binary_opening(mask_array, iterations=3)

# Convert the numpy array back to a PIL image
opened_mask_image = Image.fromarray(opened_mask_array)

# Save the opened mask image
opened_mask_image.save('outputs/opened_mask_image.png')