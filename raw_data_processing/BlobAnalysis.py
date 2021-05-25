import cv2
from copy import copy


def fill_holes(BW_image):
    BW_copy = copy(BW_image)
    contours, hierarchy = cv2.findContours(BW_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = copy(BW_image)
    filled_image[:,:] = 0
    for contour in contours:
        cv2.drawContours(filled_image, [contour], 0, 1, -1)
    return filled_image


def make_rectangle_mask(black_image,rect):
    cv2.fillPoly(black_image,[rect.reshape(-1,2)],1,1)
    return black_image







