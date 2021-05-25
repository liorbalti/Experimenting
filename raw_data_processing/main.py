import cv2
from raw_data_processing import TagsInfo as TI, Experiment, Frame
import numpy as np
from os import sep as sep
import pandas as pd
import collections

expName = '11_140720'
path = r'Y:\Lior&Einav\Experiments\experiment'+expName+'\with food'
outpath = path + sep + 'blob analysis normalized by white paper'
ex1 = Experiment.Experiment2Colors(path, get_timestamps=True)
trim_coordinates, frame_sizes, transformation_matrix = ex1.get_transformation_parameters()
# norm_mats = dict()
# norm_mats['BLGF'] = np.loadtxt(open(r"Y:\Lior&Einav\Calibration\illumination 1_5_19\NormMatrix_atto.csv", "rb"), delimiter=",",dtype='float32')
# norm_mats['GLRF'] = np.loadtxt(open(r"Y:\Lior&Einav\Calibration\illumination 1_5_19\NormMatrix_rhod.csv", "rb"), delimiter=",",dtype='float32')
#
#
# bg_masks = ex1.define_background_to_remove_from_fluorescence_frame(trim_coordinates,100)
# # food_source_masks = ex1.get_food_source_masks(trim_coordinates, frame_sizes, transformation_matrix)

norm_mats = ex1.get_fluo_background_from_paper()
norm_mats['BLGF'] = norm_mats['yellow']
norm_mats['GLRF'] = norm_mats['red']
bg_images = ex1.get_fluo_background_from_min_filter()
bg_mask = ex1.get_bg_mask_from_bg_images(bg_images)


# Tag data
print('getting tag data')
ants_info = TI.AntsInfo(path)
ants_info_all_frames = ants_info.parse_tag_data()
angle_corrections = ants_info.get_angle_corrections()
ant_parameters_list = ants_info.make_ant_parameters_list()


# definitions for video writer:
fourcc = cv2.VideoWriter_fourcc(*'XVID')
font = cv2.FONT_HERSHEY_SIMPLEX
outMovie=cv2.VideoWriter(outpath + sep + 'analyzed movie.avi',fourcc,20.0,(frame_sizes['Fluorescence'][1],frame_sizes['Fluorescence'][0]),True)

ex1.sync_cameras_go_to_start()
#ex1.go_to_frame(8888)

print('start analysis')
available_frame_to_read = True
output_dataframe = pd.DataFrame()
frame_list = []
time_list = []
acquisition_list = []
while available_frame_to_read:

    current_frame = ex1.TagsReader.frame_index
    print('frame ' + str(current_frame) + ' out of ' + str(ex1.total_number_of_frames))
    timestamp = ex1.TagsReader.timestamps[current_frame]

    if current_frame not in ex1.missing_frames:
        try:
            tag_frame, fluo_frame, output_frame, available_frame_to_read, acquisition, th = \
                ex1.create_combined_frame(trim_coordinates, frame_sizes, transformation_matrix, bg_mask, norm_mats)
        except:
            break
        cv2.putText(output_frame.overlayed_image, acquisition, (50, 210), font, 2, (255, 255, 255), 5)

    else:
        available_frame_to_read, tag_image = ex1.TagsReader.read_frame()
        tag_frame = Frame.TagsFrame(tag_image)
        tag_frame.full_transform(trim_coordinates,frame_sizes,transformation_matrix)
        output_frame = Frame.OutputFrame(tag_frame)
        cv2.putText(tag_frame.frame, 'missing fluorescence frame', (50, 210), font, 2, (255, 255, 255), 5)
        print('frame ' + str(current_frame) + ' missing from lower camera')
        acquisition = 'missing_frame'

    # output_frame.insert_food_source_mask(food_source_masks)
    cv2.putText(output_frame.overlayed_image, 'f= ' + str(current_frame), (50,70), font, 2, (255,255,255), 5)
    cv2.putText(output_frame.overlayed_image, 't= ' + '{0:3.1f}'.format(timestamp) + 's', (50, 140), font, 2, (255,255,255), 5)

    tag_frame_info = ants_info_all_frames[current_frame]

    output_line = collections.OrderedDict()
    frame_list.extend([current_frame])
    time_list.extend([timestamp])
    acquisition_list.extend([acquisition])
    for ant in tag_frame_info:
        tag = ant.id
        ant.angle_correction = float(angle_corrections[tag])
        ant.transform_coordinates(trim_coordinates, frame_sizes, transformation_matrix)
        rect = ant.make_rectangle(ant_parameters_list[tag])
        cv2.polylines(output_frame.overlayed_image,np.int32([rect]), 1, (255,255,255), 4)

        if not ant.check_out_of_bounds(frame_sizes):
            cv2.putText(output_frame.overlayed_image, tag, (int(ant.x), int(ant.y)), font, 1.9, (0, 240, 255), 3)

            if current_frame not in ex1.missing_frames:
                around_ant = Frame.AntFluoImage(fluo_frame.frame, ant)
                # around_ant.smooth() dont smooth again if already smoothing the big image
                around_ant.binarize(th)
                around_ant.make_rectangle_mask(ant_parameters_list[tag])
                total_blob_area, total_blob_intensity, all_blobs_in_rect = around_ant.sum_blobs_in_rect()
                ant.crop_intensity = total_blob_intensity
                ant.crop_area = total_blob_area

                output_line.update(ant.to_dict(['x','y','angle','error','crop_intensity','crop_area','original_x','original_y']))

    output_dataframe = output_dataframe.append(output_line,ignore_index=True)
    outMovie.write(cv2.cvtColor(output_frame.overlayed_image, cv2.COLOR_RGB2BGR))

    # plt.figure(1)
    # output_frame.show()
    # plt.pause(0.5)

outMovie.release()
output_dataframe.insert(0,'acquisition',acquisition_list)
output_dataframe.insert(0,'time',time_list)
output_dataframe.insert(0,'frame',frame_list)
output_dataframe.to_csv(outpath + sep + 'bdata_' + expName + '.csv')

a=1




