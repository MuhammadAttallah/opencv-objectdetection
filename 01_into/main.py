import cv2
import numpy as np
import os

# change the working directory to the folder this script is in.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

haystack_img = cv2.imread('albion_farm.jpg', cv2.IMREAD_UNCHANGED)
needle_img = cv2.imread('albion_cabbage.jpg', cv2.IMREAD_UNCHANGED)

# there are 6 comparison methods to choose from:
# TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
result = cv2.matchTemplate(haystack_img, needle_img, cv2.TM_CCOEFF_NORMED)

# get the best match position from the match result.
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# the max location will contain the upper left corner pixel position for the area
# that most closely matches our needle image. The max value gives an indication
# of how similar that find is to the original needle, where 1 is perfect and -1
# is exact opposite.
print('Best match top left position: %s' % str(max_loc))
print('Best match confidence: %s' % max_val)

# if the best match value is greater than 0.8, we'll trust that we found a match
threshold = 0.8
if max_val >= threshold:
    print('Found needle')

    # get dimensions of the needle image
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    # calculate the bottom right corner of the rectangle to draw
    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

    # draw a rectangle on our screenshot to highlight where we found the needle.
    # the line color can be set as an RGB tuple
    cv2.rectangle(haystack_img, top_left, bottom_right, color=(0, 255, 0),
                  thickness=2, lineType=cv2.LINE_4)

    # View the result
    #cv2.imshow('Result', haystack_img)
    #cv2.waitKey()

    # save the result image
    cv2.imwrite('result.jpg', haystack_img)
else:
    ('Needle not found')



