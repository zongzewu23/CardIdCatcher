# Import necessary libraries
import numpy as np
import cv2
import utils
import argparse
from imutils import contours

# Parse command-line arguments for input image and template image paths
ap = argparse.ArgumentParser()
ap.add_argument("-i", "-image", required=True, help="Path to the input image containing card numbers")
ap.add_argument("-t", "--template", required=True, help="Path to the template OCR-A image (digit reference)")
args = vars(ap.parse_args())

# Define a dictionary to map the first digit of a card number to its type
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

# Function to display an image in a window (for debugging and visualization)
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the template image (OCR-A digit reference) and display it
img = cv2.imread(args["template"])
cv_show("img", img)

# Convert the template image to grayscale for processing
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref', ref)

# Apply thresholding to binarize the grayscale image
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref', ref)

# Find contours of the digits in the template image
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)  # Draw contours on the template image
cv_show('img', img)

# Print the number of contours found
print(np.array(refCnts).shape)

# Sort the contours left-to-right and initialize a dictionary for storing digit ROIs
refCnts = utils.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# Extract each digit's region of interest (ROI) and resize it to a fixed size
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y+h, x:x+w]
    roi = cv2.resize(roi, (57, 58))  # Resize ROI to 57x58 pixels
    digits[i] = roi  # Store the digit ROI in the dictionary

# Define kernels for morphological operations
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # Rectangular kernel
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))    # Square kernel

# Load the input image containing card numbers and resize it for consistency
image = cv2.imread(args["image"])
cv_show('image', image)
image = utils.resize(image, width=300)

# Convert the resized image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)

# Apply a top-hat morphological operation to highlight brighter regions on a dark background
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat', tophat)

# Compute the gradient in the x-direction to emphasize vertical edges
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
gradX = np.absolute(gradX)  # Take the absolute value of the gradient
(minVal, maxVal) = (np.min(gradX), np.max(gradX))  # Find min and max gradient values
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")  # Normalize gradient values



# Print the shape of the gradient image and display it
print(np.array(gradX).shape)
cv_show('gradX', gradX)

# Apply a morphological closing operation to connect small gaps in the gradient image
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX', gradX)

# Threshold the gradient image to create a binary image
# Use OTSU's method to automatically determine the threshold value
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)

# Perform another morphological closing operation to further connect gaps in the binary image
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv_show('thresh', thresh)

# Find contours in the thresholded image to identify potential regions of interest
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts

# Create a copy of the original image to draw contours on it for visualization
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)  # Draw contours in red
cv_show('img', cur_img)

# Initialize a list to store bounding boxes of regions of interest
locs = []

# Loop through each contour to filter and store valid bounding boxes
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)  # Get bounding box coordinates
    ar = w / float(h)  # Calculate the aspect ratio of the bounding box

    # Check if the bounding box meets the criteria for a valid region of interest
    if 2.5 < ar < 4.0:  # Aspect ratio range
        if 40 < w < 55 and 10 < h < 20:  # Width and height range
            locs.append((x, y, w, h))  # Add the bounding box to the list

# Sort the bounding boxes from left to right based on the x-coordinate
locs = sorted(locs, key=lambda x: x[0])
output = []

# Loop through each region of interest
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []  # Initialize a list to store the recognized digits in the group
    group = gray[gY-5:gY+gH+5, gX-5:gX+gW+5]  # Extract the region of interest with padding
    cv_show('group', group)

    # Threshold the extracted region to prepare it for digit contour detection
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)

    # Find contours of digits in the extracted region
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]  # Sort digits left to right

    # Loop through each detected digit contour
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)  # Get bounding box for the digit
        roi = group[y:y+h, x:x+w]  # Extract the digit region of interest
        roi = cv2.resize(roi, (57, 58))  # Resize the digit to match the template size
        cv_show('roi', roi)

        scores = []  # List to store template matching scores

        # Compare the extracted digit with each template digit using template matching
        for (digit, digiROI) in digits.items():
            result = cv2.matchTemplate(roi, digiROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)  # Store the matching score

        # Find the digit with the highest matching score and add it to the group output
        groupOutput.append(str(np.argmax(scores)))

    # Draw a rectangle around the detected group and annotate it with the recognized digits
    cv2.rectangle(image, (gX-5, gY-5), (gX+gW+5, gY+gH+5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # Add the group output to the final result
    output.extend(groupOutput)

# Print the card type and recognized card number
print("Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Card ID: {}".format("".join(output)))

# Display the final annotated image with detected card details
cv2.imshow("Image", image)
cv2.waitKey(0)
