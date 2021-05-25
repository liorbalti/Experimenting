import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import cv2.cv as cv
import math
from copy import copy, deepcopy
import os
import sys
sys.path.append(r"Z:\\Lior\\Bugtag\\bugtag_new2")
# import BugTagAnal_f
# import scipy.fftpack

# Parameters


# Functions


class Formatter(object):
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)


def show_z(img, gray=0):
    fig, ax = plt.subplots()
    im = ax.imshow(img, cmap='gray', interpolation='none')
    ax.format_coord = Formatter(im)
    ax.containers
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)


def DrawSquares(imtags, sqlist):
    for sq in sqlist:
        cv2.polylines(imtags, np.array([sq], dtype=np.int32), 1, (255, 0, 0), 6)


def BlobsCentroidList(contours, areaThres, file_name, frame):
    c_list = [frame]
    for i in np.arange(len(contours)):
        m = cv2.moments(contours[i])
        if m['m00'] > areaThres:
            c_list += [m['m00'], int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])]
    with open(file_name, 'a') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerows([c_list])


def writeSqureList(squreList, file_name, frame, transM):
    # midel of the squers
    s = np.array(squreList)
    l = s[:, :, :].sum(axis=1) / 4
    l = np.append(l, np.ones((len(l), 1)), 1)
    tag_cent = np.dot(transM, l.transpose())
    with open(file_name, 'a') as r:
        writer2 = csv.writer(r, delimiter=',', lineterminator='\n')
        writer2.writerows([['frame #' + str(frame)]])
        # writer2.writerows(l)
        writer2.writerows([tag_cent[0]])
        writer2.writerows([tag_cent[1]])


def writedata(file_name, array):
    with open(file_name, 'a') as g:
        writer3 = csv.writer(g, delimiter=',', lineterminator='\n')
        writer3.writerows(array)


def RhodPicTransform(image,angle1,rhod_trim):
    # angle1 = 2.3
    # rhod_trim = [176, 2473, 9, 1959]
    Mrot_f = cv2.getRotationMatrix2D((len(image) / 2, len(image[:][:][0]) / 2), angle1, 1)
    image = cv2.flip(image, 0)  # flip
    image = cv2.warpAffine(image, Mrot_f, (3000, 3000))  # rotate
    image = image[rhod_trim[2]:rhod_trim[3], rhod_trim[0]:rhod_trim[1]]  # crop
    del angle1, rhod_trim, Mrot_f
    return image


def TagPicTransform(image,angle2,tag_trim):
    # angle2 = -88.1
    # tag_trim = [63, 4963, 555, 4724]
    Mrot = cv2.getRotationMatrix2D((5120 / 2, 5120 / 2), angle2, 1)
    image = cv2.warpAffine(image, Mrot, (5750, 5750))  # rotate
    image = image[tag_trim[2]:tag_trim[3], tag_trim[0]:tag_trim[1]]  # crop
    del angle2, tag_trim, Mrot
    return image, Mrot

def separate_coordinates(coordinates):
    xs = []
    ys = []
    for x in coordinates:
        xs.append(x[0])
        ys.append(x[1])
    return xs, ys





# Finding Parameters

cap_tag = cv2.VideoCapture(r"Z:\Lior&Einav\Small nest 31_01_19\GT6600_31-01-19_16.44.51exp11ms.avi")
ret1,imtags_temp = cap_tag.read()

rhod_temp = cv2.imread(r"Z:\Lior&Einav\Small nest 31_01_19\UI359x_31-01-19_16.44.51exp33ms\UI359x_31-01-19_16.44.51exp33msBLGF.tif")


# Trim lower camera image
answer = True
while answer:

    plt.imshow(imtags_temp)
    plt.pause(2)

    print("click on the 4 corners to get their coordinates")
    corners = plt.ginput(4)

    [xs, ys] = separate_coordinates(corners)

    tag_trim = [math.floor(min(xs)),math.ceil(max(xs)),math.floor(min(ys)),math.ceil(max(ys))]
    imtags_temp2=imtags_temp[tag_trim[2]:tag_trim[3],tag_trim[0]:tag_trim[1]]
    plt.close()
    plt.imshow(imtags_temp2)
    plt.pause(2)

    user_answer=input('try again? yes[1]/no[0]')
    if user_answer =='0':
        answer = False
    elif user_answer =='1':
        answer = True
    else:
        print('Not acceptable answer. Trying again.')
        answer = True
