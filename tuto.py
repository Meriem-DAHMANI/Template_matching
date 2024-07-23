import numpy as np
import cv2

def load_image(image_path, grayscale=True):
    """
    Load an image from a given path.
    
    Args:
    image_path (str): The path to the image to be loaded.
    grayscale (bool): If True, loads the image in grayscale. Default is True.
    
    Returns:
    image: The loaded image.
    """
    if grayscale:
        return cv2.imread(image_path, 0)
    return cv2.imread(image_path)

def apply_template_matching(img, template, method):
    """
    Apply template matching on a given image with a specified template and method.
    
    Args:
    img (array): The image to search in.
    template (array): The template to search for.
    method (int): The template matching method from OpenCV.
    
    Returns:
    result: The result of the template matching.
    min_val: The minimum value found by cv2.minMaxLoc.
    max_val: The maximum value found by cv2.minMaxLoc.
    min_loc: The location of the minimum value.
    max_loc: The location of the maximum value.
    """
    result = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return result, min_val, max_val, min_loc, max_loc

def draw_rectangle(img, location, template_size):
    """
    Draw a rectangle on an image.
    
    Args:
    img (array): The image on which to draw the rectangle.
    location (tuple): The (x, y) location of the top-left corner of the rectangle.
    template_size (tuple): The size (height, width) of the template.
    
    Returns:
    img_with_rectangle: The image with a rectangle drawn around the matching region.
    """
    h, w = template_size
    bottom_right = (location[0] + w, location[1] + h)
    img_with_rectangle = cv2.rectangle(img.copy(), location, bottom_right, 255, 5)
    return img_with_rectangle

def main():
    # Load the images
    img = load_image('assets/soccer_practice.jpg')  # Load the image in grayscale
    template = load_image('assets/ball.png')  # Load the template in grayscale
    h, w = template.shape  # Get the size of the template

    # List of comparison methods
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
               cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    for method in methods:
        # Apply template matching
        result, min_val, max_val, min_loc, max_loc = apply_template_matching(img, template, method)
        
        # Determine the best match location
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            location = min_loc
        else:
            location = max_loc

        # Draw a rectangle around the matching region
        img_with_rectangle = draw_rectangle(img, location, (h, w))
        
        # Display the image with the rectangle
        cv2.imshow(f'Match - Method: {method}', img_with_rectangle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
