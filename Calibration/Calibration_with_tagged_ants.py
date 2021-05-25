import cv2
from raw_data_processing import TagsInfo as TI, Experiment, Frame
import numpy as np
import helper_functions as hf
import pandas as pd
from os import sep as sep

norm_mat = np.loadtxt(open(r"Z:\Lior&Einav\Calibration\videos\illumination 1_5_19\NormMatrix_rhod.csv", "rb"),
                      delimiter=",", dtype='float32')

feedings = [1,2,3,4,5,6]
for f in feedings:
    path = r'P:\Lior & Einav\Ant Calibration 25_6_19\Rhod\feed' + str(f)
    outpath = path + sep + 'blob analysis normalized'
    ex1 = Experiment.ExperimentOneColor(path)
    trim_coordinates, frame_sizes, transformation_matrix = ex1.get_transformation_parameters()
    bg_masks = ex1.define_background_to_remove_from_fluorescence_frame(trim_coordinates,20, norm_mat=norm_mat)


    ants_info = TI.AntsInfo(path)
    ants_info_all_frames = ants_info.parse_tag_data()
    angle_corrections = ants_info.get_angle_corrections()
    ant_parameters_list = ants_info.make_ant_parameters_list()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    font = cv2.FONT_HERSHEY_SIMPLEX
    outMovie=cv2.VideoWriter(outpath + sep + 'analyzed movie.avi',fourcc,20.0,(frame_sizes['Fluorescence'][1],frame_sizes['Fluorescence'][0]),True)

    ex1.sync_cameras_go_to_start()

    th, color, acquisition = ex1.set_binarize_parameters()

    available_frame_to_read_tags = True
    available_frame_to_read_fluo = True
    intensity_df = pd.DataFrame()
    area_df = pd.DataFrame()
    ant_movies = dict()
    recognized_ants = []

    # ex1.go_to_frame(200)

    while available_frame_to_read_tags and available_frame_to_read_fluo:

        current_frame = ex1.TagsReader.frame_index
        # timestamp = ex1.TagsReader.timestamps[current_frame]
        print('frame ' + str(current_frame))

        try:
            available_frame_to_read_tags, tag_image = ex1.TagsReader.read_frame()
        except:
            print("error in tag read frame. frame " + str(current_frame) + ". break from loop")
            break
        tag_frame = Frame.TagsFrame(tag_image)
        tag_frame.full_transform(trim_coordinates,frame_sizes,transformation_matrix)

        if current_frame in ex1.missing_frames_tags:
            available_frame_to_read_fluo, _ = ex1.FluoReader.read_frame()

        if current_frame not in ex1.missing_frames:
            try:
                available_frame_to_read_fluo, fluo_image = ex1.FluoReader.read_frame()
            except:
                print("error in fluo read frame. frame " + str(current_frame) + ". break from loop")
                break
            fluo_frame = Frame.FluorescenceFrame(fluo_image)
            fluo_frame.full_transform(trim_coordinates,th,bg_masks[acquisition],norm_mat=norm_mat)

            output_frame = Frame.OutputFrame(tag_frame, fluo_frame)
            output_frame.overlay(color=color)

        else:
            print('missing fluorescence frame ' + str(current_frame))
            output_frame = Frame.OutputFrame(tag_frame)
            cv2.putText(tag_frame.frame, 'missing fluorescence frame', (50, 210), font, 2, (255, 255, 255), 5)

        cv2.putText(output_frame.overlayed_image, 'f= ' + str(current_frame), (50,70), font, 2, (255,255,255), 5)

        tag_frame_info = ants_info_all_frames[current_frame]

        intensity_line = dict()
        area_line = dict()
        for ant in tag_frame_info:
            tag = ant.id
            ant.angle_correction = float(angle_corrections[tag])
            ant.transform_coordinates(trim_coordinates, frame_sizes, transformation_matrix)
            rect = ant.make_rectangle(ant_parameters_list[tag])
            cv2.polylines(output_frame.overlayed_image,np.int32([rect]), 1, (255,255,255), 4)

            if not ant.check_out_of_bounds(frame_sizes):
                cv2.putText(output_frame.overlayed_image, tag, (int(ant.x), int(ant.y)), font, 1.9, (0, 240, 255), 3)

                if tag not in recognized_ants:
                    recognized_ants.extend([tag])
                    ant_movies[tag] = cv2.VideoWriter(outpath + sep + tag + '.avi',fourcc,20.0,(400,400),True)

                if current_frame not in ex1.missing_frames:
                    around_ant = Frame.AntFluoImage(fluo_frame.frame, ant)
                    around_ant.smooth()
                    around_ant.binarize(th)
                    around_ant.make_rectangle_mask(ant_parameters_list[tag])
                    total_blob_area, total_blob_intensity, all_blobs_in_rect = around_ant.sum_blobs_in_rect()
                    ant.crop_intensity = total_blob_intensity
                    ant.crop_area = total_blob_area
                    ant_with_blob = hf.imoverlay(around_ant.frame*3,all_blobs_in_rect,(220, 20, 60))
                    cv2.putText(ant_with_blob, 'f= ' + str(current_frame), (50, 70), font, 2, (255, 255, 255), 5)

                    # show current ant with her considered blobs
                    # plt.figure(2)
                    # plt.imshow(ant_with_blob)
                    # plt.pause(1)

                    if ant_with_blob.shape != (400,400,3):
                        ant_with_blob = cv2.copyMakeBorder(ant_with_blob,0,400-ant_with_blob.shape[0],0,400-ant_with_blob.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0))
                    ant_movies[tag].write(ant_with_blob)
                    intensity_line['a' + ant.id] = total_blob_intensity
                    area_line['a' + ant.id] = total_blob_area

        intensity_df = intensity_df.append(intensity_line,ignore_index=True)
        area_df = area_df.append(area_line,ignore_index=True)

        # plt.figure(1)
        # output_frame.show()
        # plt.pause(0.5)
        outMovie.write(cv2.cvtColor(output_frame.overlayed_image,cv2.COLOR_RGB2BGR))

    outMovie.release()
    intensity_df.to_csv(outpath+sep+'blob_total_intensity.csv')
    area_df.to_csv(outpath+sep+'blob_area.csv')
    print('done!')

a = 1
