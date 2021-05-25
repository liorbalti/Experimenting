import cv2
import matplotlib.pyplot as plt
from raw_data_processing import TagsInfo as TI, Experiment, Frame
import numpy as np

path = r'Z:\Lior&Einav\Experiments\Experiment_2_21_4_19\With food'
ex1 = Experiment.Experiment2Colors(path)
ants_info = TI.AntsInfo(path)
ants_info_all_frames = ants_info.parse_tag_data()
angle_corrections = ants_info.get_angle_corrections()
ant_parameters_list = ants_info.make_ant_parameters_list()

trim_coordinates, frame_sizes, transformation_matrix = ex1.get_transformation_parameters()
bg_masks = ex1.define_background_to_remove_from_fluorescence_frame(trim_coordinates,20)
food_source_masks = ex1.get_food_source_masks(trim_coordinates, frame_sizes, transformation_matrix)

# definitions for video writer:
fourcc = cv2.VideoWriter_fourcc(*'XVID')
font = cv2.FONT_HERSHEY_SIMPLEX
# outMovie=cv2.VideoWriter(r'D:\Lior\phd\temp_output\combined video attempt\comb15_12_16_manipulation.avi',fourcc,20.0,frame_sizes['Fluorescence'],True)

ex1.sync_cameras_go_to_start()
# ex1.go_to_frame(1500)

available_frame_to_read = True
while available_frame_to_read:

    current_frame = ex1.TagsReader.frame_index
    timestamp = ex1.TagsReader.timestamps[current_frame]

    available_frame_to_read, tag_image = ex1.TagsReader.read_frame()
    tag_frame = Frame.TagsFrame(tag_image)
    tag_frame.full_transform(trim_coordinates,frame_sizes,transformation_matrix)

    if current_frame not in ex1.missing_frames:
        th, color, acquisition = ex1.set_binarize_parameters()

        _, fluo_image = ex1.FluoReader.read_frame()
        fluo_frame = Frame.FluorescenceFrame(fluo_image)
        fluo_frame.full_transform(trim_coordinates,th,bg_masks[acquisition])

        output_frame = Frame.OutputFrame(tag_frame, fluo_frame)
        output_frame.overlay(color=color)

        cv2.putText(output_frame.overlayed_image, acquisition, (50, 210), font, 2, (255, 255, 255), 5)

    else:
        output_frame = Frame.OutputFrame(tag_frame)
        cv2.putText(tag_frame.frame, 'missing fluorescence frame', (50, 210), font, 2, (255, 255, 255), 5)
        print('frame ' + str(current_frame) + ' missing from lower camera')

    output_frame.insert_food_source_mask(food_source_masks)
    cv2.putText(output_frame.overlayed_image, 'f= ' + str(current_frame), (50,70), font, 2, (255,255,255), 5)
    cv2.putText(output_frame.overlayed_image, 't= ' + '{0:3.1f}'.format(timestamp) + 's', (50, 140), font, 2, (255,255,255), 5)

    tag_frame_info = ants_info_all_frames[current_frame]
    for ant in tag_frame_info:
        tag = ant.id
        ant.angle_correction = float(angle_corrections[tag])
        ant.transform_coordinates(trim_coordinates, frame_sizes, transformation_matrix)
        rect = ant.make_rectangle(ant_parameters_list[tag])
        cv2.polylines(output_frame.overlayed_image,np.int32([rect]), 1, (255,255,255), 4)
        cv2.putText(output_frame.overlayed_image, tag, (int(ant.x), int(ant.y)), font, 1.9, (0, 240, 255), 3)



        # ant_fluo_image, cropped_coor = blobs.crop_around_ant(fluo_frame.frame, ant.x, ant.y, 200, frame_sizes['Fluorescence'])
        # ant_rect = ant.make_rectangle_in_cropped_image(cropped_coor['x'],cropped_coor['y'],ant_parameters_list[tag])
        # rect_mask = blobs.make_rectangle_mask(np.zeros_like(ant_fluo_mask),ant_rect)



        # ant_fluo_overlay = hf.imoverlay(ant_fluo_image, ant_fluo_mask, (255, 0, 0))
        # plt.imshow(ant_fluo_overlay)

    plt.figure(1)
    output_frame.show()
    plt.pause(0.5)

a=1




