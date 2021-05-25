import cv2
import csv
from datetime import datetime as dt
import os
import numpy as np
from contextlib import contextmanager
import sys


def separate_coordinates(coordinates):
    xs = []
    ys = []
    for x in coordinates:
        xs.append(x[0])
        ys.append(x[1])
    return xs, ys


def write_to_csv(path, csvData):
    with open(path,'w',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()


def read_from_csv(path):
    try:
        with open(path,'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            all_rows = []
            for row in csv_reader:
                all_rows.append(row)
    except FileNotFoundError:
        all_rows = None
    return all_rows


def cumulative_list(list_in):
    return [sum(list_in[0:x]) for x in range(0, len(list_in)+1)]


def datenum(d):
    return 366 + d.toordinal() + (d - dt.fromordinal(d.toordinal())).total_seconds()/(24*60*60)


def check_if_file_exists(directory,filename):
    exists = os.path.isfile(directory + os.sep + filename)
    return exists


def imoverlay(image,mask,color_rgb):
    color_screen = np.zeros(mask.shape + (3,), mask.dtype)
    color_screen[:, :] = color_rgb
    color_mask = cv2.bitwise_and(color_screen, color_screen, mask=np.uint8(mask))
    if len(image.shape) == 2:  # if image is grayscale, convert to rgb
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    output_image = cv2.addWeighted(color_mask, 1, image, 1, 0, image, dtype=32)
    return output_image


def make_rectangle(x,y,angle,width,height,distance_from_point):
    dsin = np.sin(angle * np.pi / 180)
    dcos = np.cos(angle * np.pi / 180)
    rect = np.array([[x + width * dsin - distance_from_point * dcos, y - width * dcos - distance_from_point * dsin],
                     [x + width * dsin - height * dcos, y - width * dcos - height * dsin],
                     [x - width * dsin - height * dcos, y + width * dcos - height * dsin],
                     [x - width * dsin - distance_from_point * dcos, y + width * dcos - distance_from_point * dsin]],
                    dtype=np.int32)
    return rect


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

