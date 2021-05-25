import cv2
import numpy as np


def read_tag_file(full_file_name):
    tags_file = np.genfromtxt(full_file_name,delimiter=',',skip_header=1,dtype=np.float32)
    return tags_file


def read_line(tags_file,line):
    taginfo = tags_file[line]
    return taginfo


def get_tag_list(full_file_name):
    f = open(full_file_name, 'r')
    tags_line = f.read_line()
    tags = tags_line.split(',')
    return tags


def put_tags_on_image(image,taginfo,tags):
    for idx in np.arange(2,taginfo.shape[0],4):
        if taginfo[idx]>=0:
            cv2.putText(image,str(tags[idx][0:-2]),(taginfo[idx],taginfo[idx+1]),font,1,(0,165,255),2)
    return image

