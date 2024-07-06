from PIL import Image, ImageEnhance

def ig_filter(image_path):
    original_image = Image.open(image_path)
    original_image.show(title="Original Image")

    brightness = ImageEnhance.Brightness(original_image)
    reduced_brightness_image = brightness.enhance(0.5)

    contrast = ImageEnhance.Contrast(reduced_brightness_image)
    increased_contrast_image = contrast.enhance(1.5)

    saturation = ImageEnhance.Color(increased_contrast_image)
    final_filtered_image = saturation.enhance(1.5)
    
    return final_filtered_image

# Example usage:
input_image_path = "image1.jpg"

output_image = ig_filter(input_image_path)
output_image.show()
