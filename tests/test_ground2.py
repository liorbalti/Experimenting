import cv2
from raw_data_processing import TagsInfo as TI, Experiment, Frame
import numpy as np
from os import sep as sep
import pandas as pd
import collections
import matplotlib.pyplot as plt

expName = '11_140720'
path = r'Y:\Lior&Einav\Experiments\experiment'+expName+'\with food'
ex1 = Experiment.Experiment2Colors(path, get_timestamps=True)
trim_coordinates, frame_sizes, transformation_matrix = ex1.get_transformation_parameters()

norm_mats = ex1.get_fluo_background_from_paper()
norm_mats['BLGF'] = norm_mats['yellow']
norm_mats['GLRF'] = norm_mats['red']
bg_images = ex1.get_fluo_background_from_min_filter()
bg_mask = ex1.get_bg_mask_from_bg_images(bg_images)

tag_frame, fluo_frame, output_frame, available_frame_to_read, acquisition, th = \
                ex1.create_combined_frame(trim_coordinates, frame_sizes, transformation_matrix, bg_mask, norm_mats)

output_frame.show()

# fig, axs = plt.subplots(1,2)
# axs[0].imshow(norm_mats['red'])
# axs[1].imshow(norm_mats['yellow'])
# axs[0].set_title('red')
# axs[1].set_title('yellow')
#
# fig2, axs2 = plt.subplots(1,2)
# axs2[0].imshow(bg_images['red'])
# axs2[1].imshow(bg_images['yellow'])
# axs2[0].set_title('red')
# axs2[1].set_title('yellow')
#
# fig3, axs3 = plt.subplots()
# axs3.imshow(bg_mask)

a=1