import cv2
import matplotlib.pyplot as plt
import math
import helper_functions as hf
from copy import copy
from raw_data_processing import BlobAnalysis as blobs, TagsInfo
import numpy as np
from skimage import morphology
from skimage import exposure


class Frame:
    def __init__(self, frame):
        self.frame = frame
        self.original_frame = frame

    def trim_frame(self, trim_coordinates):
        return self.frame[trim_coordinates[2]:trim_coordinates[3], trim_coordinates[0]:trim_coordinates[1]]

    def get_trim_coordinates(self):

        while True:
            f1 = plt.figure()
            plt.imshow(self.frame)

            print("click on the 4 corners to get their coordinates")
            corners = plt.ginput(4, timeout=-1)

            [xs, ys] = hf.separate_coordinates(corners)

            trim_coordinates = [math.floor(min(xs)), math.ceil(max(xs)), math.floor(min(ys)), math.ceil(max(ys))]
            trimmed_frame = self.trim_frame(trim_coordinates)
            plt.close(f1)

            f2 = plt.figure()
            plt.imshow(trimmed_frame)
            plt.pause(2)
            user_answer = input('Enter if everything is OK, input anything else to redo')
            plt.close(f2)
            if user_answer is '':
                break
        plt.close('all')
        return trim_coordinates, trimmed_frame.shape


class TagsFrame(Frame):
    def __init__(self, frame):
        super().__init__(frame)

    def flip(self):
        self.frame = cv2.flip(self.frame, 1)

    def transform(self, transformation_matrix, size):
        self.frame = cv2.warpAffine(self.frame, transformation_matrix, size)

    def full_transform(self, trim_coordinates, frame_sizes, transformation_matrix, adjust=True, max_in=70):
        self.frame = self.trim_frame(trim_coordinates['Tags'])
        self.flip()
        self.transform(transformation_matrix, (frame_sizes['Fluorescence'][1], frame_sizes['Fluorescence'][0]))
        if adjust:
            self.adjust_intensity(max_in)

    def adjust_intensity(self, max_in):
        self.frame = exposure.rescale_intensity(self.frame,(0,max_in),(0,255)).astype('uint8')


class FluorescenceFrame(Frame):
    def __init__(self, frame):
        super().__init__(frame)
        # self.frame = cv2.cvtColor(self.frame,cv2.COLOR_RGB2GRAY)
        self.contours = None
        self.mask = None

    def binarize(self, threshold, min_blob_size=5):
        _, BW = cv2.threshold(self.frame, threshold, 255, cv2.THRESH_BINARY)
        # self.contours = cv2.findContours(BW,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cleaned_BW = morphology.remove_small_objects(BW.astype(bool), min_size=min_blob_size)
        self.mask = cleaned_BW.astype(int)*255

    def full_transform(self,trim_coordinates,th,bg_mask=None,norm_mat=None,bg_image=None):
        self.normalize_illumination(norm_mat)
        self.subtract_background_image(bg_image)
        self.remove_background_mask(bg_mask)
        self.frame = self.trim_frame(trim_coordinates['Fluorescence'])
        self.smooth()
        self.binarize(threshold=th)

    def subtract_background_image(self,bg_image):
        if bg_image is not None:
            self.frame = cv2.subtract(self.frame, bg_image)
        pass

    def remove_background_mask(self,bg_mask):
        if bg_mask is not None:
            if bg_mask.dtype == 'bool':
                self.frame[bg_mask] = 0
            else:
                self.frame[bg_mask > 0] = 0

    def normalize_illumination(self,normalization_matrix):
        if normalization_matrix is not None:
            tempI = self.frame/normalization_matrix
            self.frame = np.array(tempI,dtype='float32')

    def smooth(self, d=11, sigma=20):
        smoothed = cv2.bilateralFilter(self.frame, d, sigma, sigma)
        self.frame = smoothed

    # Todo: remove_brood_roi


class AntFluoImage(FluorescenceFrame):
    def __init__(self, img, ant: TagsInfo.CurrentAntData, s=200):
        around_ant, cropped_coor = self.crop_around_ant(img, ant.x, ant.y, s)
        super().__init__(around_ant)
        self.tag_x = cropped_coor['x']
        self.tag_y = cropped_coor['y']
        self.ant = ant
        self.rect_mask = None

    def crop_around_ant(self, img, x, y, s):
        x_pixel = int(round(x))
        y_pixel = int(round(y))
        ny, nx = img.shape
        around_ant = copy(img[max(0, y_pixel - s):min(y_pixel + s, ny), max(0, x_pixel - s):min(x_pixel + s, nx)])
        cropped_coor = dict()
        for key, coor in zip(['x', 'y'], [x, y]):
            if coor - s > 0:
                cropped_coor[key] = s
            else:
                cropped_coor[key] = coor
        return around_ant, cropped_coor

    def smooth(self, n=5, filter_type='median'):
        if filter_type == 'mean':
            self.frame = cv2.blur(self.frame, (n,n))
        elif filter_type == 'median':
            self.frame = cv2.medianBlur(self.frame,n)

    def fill(self):
        self.mask = blobs.fill_holes(self.mask)

    def make_rectangle_mask(self, ant_parameters: TagsInfo.AntParameters):
        rect = hf.make_rectangle(self.tag_x, self.tag_y, self.ant.angle,ant_parameters.rectangle_size[0],
                                     ant_parameters.rectangle_size[1],ant_parameters.dist_from_tag)
        rect_mask = cv2.fillPoly(np.zeros_like(self.mask), [rect.reshape(-1, 2)], 1, 1)
        self.rect_mask = rect_mask

    def sum_blobs_in_rect(self):
        intersection = np.sum(cv2.bitwise_and(self.mask, self.mask, mask=np.uint8(self.rect_mask)))

        total_blob_area = 0
        total_blob_intensity = 0
        all_blobs_in_rect = np.zeros_like(self.mask)

        if intersection > 0:
            _, contours, hierarchy = cv2.findContours(copy(np.uint8(self.mask)), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # contours, hierarchy = cv2.findContours(copy(np.uint8(self.mask)), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                current_blob = np.zeros_like(self.mask)
                cv2.drawContours(current_blob,[contour],0,255,-1)
                intersection2 = np.sum(cv2.bitwise_and(current_blob,current_blob,mask=np.uint8(self.rect_mask)))

                if intersection2 > 0:
                    #m = cv2.moments(contour)
                    #total_blob_area += m['m00']
                    current_blob_area = np.sum(current_blob>0)
                    total_blob_area += current_blob_area
                    grayscale_blob = cv2.bitwise_and(self.frame,self.frame,mask=np.uint8(current_blob))
                    total_blob_intensity += np.sum(grayscale_blob)

                    all_blobs_in_rect += current_blob

        return total_blob_area, total_blob_intensity, all_blobs_in_rect


class OutputFrame:
    # def __init__(self,tags_frame: TagsFrame, fluo_frame=None):
    def __init__(self, tags_frame, fluo_frame=None):
        self.tags_frame = tags_frame
        self.fluo_frame = fluo_frame
        if len(tags_frame.frame.shape) == 2:
            self.overlayed_image = cv2.cvtColor(tags_frame.frame,cv2.COLOR_GRAY2RGB)
        else:
            self.overlayed_image = tags_frame.frame

    def overlay(self, color='red'):
        # TODO: check that fluo_frame is FluorescenceFrame
        if color == 'red':
            color_rgb = (220, 20, 60)
        elif color == 'yellow':
            # color_rgb = (173, 255, 47) green-yellow
            # color_rgb = (30,144,255) blue
            color_rgb = (255, 215, 0)  # gold
        self.overlayed_image = hf.imoverlay(self.overlayed_image,self.fluo_frame.mask,color_rgb)

    def insert_food_source_mask(self,food_source_masks):  # ,color):
        for color, color_rgb in zip(['red','yellow'],[(220, 20, 60),(255, 215, 0)]):
            self.overlayed_image = hf.imoverlay(self.overlayed_image,food_source_masks[color],color_rgb)

    def show(self):
        plt.imshow(self.overlayed_image)



