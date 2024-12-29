import cv2  # Import the OpenCV library for image processing

def sort_contours(cnts, method="left-to-right"):
    """
    Sorts contours based on a specified method.

    Parameters:
        cnts (list): A list of contours to be sorted.
        method (str): Sorting method, default is "left-to-right".
            Options:
            - "left-to-right": Sort contours from left to right.
            - "right-to-left": Sort contours from right to left.
            - "top-to-bottom": Sort contours from top to bottom.
            - "bottom-to-top": Sort contours from bottom to top.

    Returns:
        cnts (list): Sorted contours.
        boundingBoxes (list): Sorted bounding box coordinates corresponding to the contours.
    """
    reverse = False  # Determines if sorting should be in reverse order
    i = 0  # Determines the axis to sort by: 0 for x-coordinate, 1 for y-coordinate

    # If sorting is "right-to-left" or "bottom-to-top", set reverse to True
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    
    # If sorting is "top-to-bottom" or "bottom-to-top", set axis to y-coordinate (1)
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # Compute bounding boxes for each contour
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # Sort contours and bounding boxes based on the specified axis and order
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
    )

    return cnts, boundingBoxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resizes an image to the specified width or height while maintaining aspect ratio.

    Parameters:
        image (numpy.ndarray): The input image to be resized.
        width (int, optional): The desired width of the output image. Defaults to None.
        height (int, optional): The desired height of the output image. Defaults to None.
        inter (int, optional): Interpolation method for resizing. Defaults to cv2.INTER_AREA.

    Returns:
        numpy.ndarray: The resized image.
    """
    dim = None  # Dimensions for the resized image
    (h, w) = image.shape[:2]  # Get the height and width of the input image

    # If both width and height are None, return the original image
    if width is None and height is None:
        return image

    # If width is None, calculate the scale factor based on height
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:  # If height is None, calculate the scale factor based on width
        r = width / float(w)
        dim = (width, int(h * r))

    # Perform the resizing operation
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized
