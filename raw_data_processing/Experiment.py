from os import sep as sep
from raw_data_processing import Reader, Frame
import matplotlib.pyplot as plt
import cv2
import numpy as np
import csv
import pandas as pd
import helper_functions as hf
import pickle
import math
import scipy.io as sio
from skimage import morphology


class Experiment:
    def __init__(self, path, fluoReader_type='vid', get_timestamps=False, tags_vid_prefix='GT'):
        self.root_path = path
        self.path = path
        self.FluorescencePath = path
        self.TagsPath = path
        if fluoReader_type == 'vid':
            self.FluoReader = Reader.VideoReader(path, prefix='UI')
        elif fluoReader_type == 'img':
            self.FluoReader = Reader.ImageReader(path, prefix='UI')
        else:
            raise ValueError('invalid fluoReader_type: ' + fluoReader_type + '. Must be "vid" or "img".')
        self.TagsReader = Reader.VideoReader(path, prefix=tags_vid_prefix, get_timestamps=get_timestamps)
        self.missing_frames = self.read_missing_frames_file()
        self.missing_frames_tags = self.read_missing_frames_file(tags=True)
        self.parameters = self.read_parameters_file_to_dict()
        self.sync_cameras_go_to_start()
        self.total_number_of_frames = self.TagsReader.cumulative_video_lengths[-1]-self.parameters['tag_cam_frame_skip']

    def read_parameters_file_to_dict(self):
        full_path = self.root_path + sep + 'Parameters.csv'
        parameters_df = pd.read_csv(full_path)
        parameters_list = parameters_df.to_dict('records')
        return parameters_list[0]

    def set_binarize_parameters(self):
        try:
            acquisition = self.FluoReader.acquisition
        except AttributeError:
            acquisition = 'no acquisition'
        if acquisition == 'BLGF':
            th = int(self.parameters['Yellow_threshold'])
            color = 'yellow'
        elif acquisition == 'GLRF':
            th = int(self.parameters['Red_threshold'])
            color = 'red'
        else:  # reader is video
            th = int(self.parameters['threshold'])
            color = 'red'
            acquisition = self.parameters['acquisition']
        return th, color, acquisition

    def define_fluorescence_background_by_threshold(self, trim_coordinates, norm_mat=None):
        th, color, acquisition = self.set_binarize_parameters()
        _, fluo_image = self.FluoReader.read_frame()
        fluo_frame = Frame.FluorescenceFrame(fluo_image)
        if norm_mat is not None:
            fluo_frame.normalize_illumination(norm_mat)
        fluo_frame.frame = fluo_frame.trim_frame(trim_coordinates['Fluorescence'])
        fluo_frame.binarize(threshold=th - 5)
        output_frame = Frame.OutputFrame(fluo_frame, fluo_frame)
        output_frame.overlay()
        output_frame.show()
        plt.pause(2)
        print("define ants area to remove from background mask")
        pts = plt.ginput(4,timeout=-1)
        [xs, ys] = hf.separate_coordinates(pts)
        coordinates = [math.floor(min(xs)), math.ceil(max(xs)), math.floor(min(ys)), math.ceil(max(ys))]
        fluo_frame.mask[coordinates[2]:coordinates[3], coordinates[0]:coordinates[1]] = 0
        fluo_frame.mask = cv2.dilate(fluo_frame.mask, np.ones((7, 7)), iterations=1)
        plt.imshow(fluo_frame.mask)
        plt.pause(2)
        return fluo_frame.mask, acquisition

    def get_fluo_background_from_paper(self, matfile_path='Y:\Lior&Einav\Matlab\Experiments2',
                                      paper_path=r'Y:\Lior&Einav\white_paper_031120\UI359x_03-11-20_15.03.52.102exp33ms'):
        white_paper = {}
        for c in ['red','yellow']:
            matfile_name = 'white_paper_norm_mat_' + c + '.mat'
            m_dict = sio.loadmat(matfile_path+sep+matfile_name)
            white_paper[c] = m_dict['norm_mat_'+c]
        return white_paper

        # acquisitions = ['BLGF', 'GLRF']
        # acquisition_dict = {'BLGF': 'red', 'GLRF': 'yellow'}
        #
        # white_paper = {}
        # for a in acquisitions:
        #     R = Reader.ImageReader(paper_path,extension=a+'.tif')
        #     white_paper[acquisition_dict[a]] = R.get_average_image()
        #
        # return white_paper

    def get_fluo_background_from_min_filter(self):
        acquisitions = ['BLGF', 'GLRF']
        acquisition_dict = {'BLGF': 'red', 'GLRF': 'yellow'}

        min_background = {}
        for a in acquisitions:
            R = Reader.ImageReader(self.FluorescencePath,extension=a+'.tif')
            min_background[acquisition_dict[a]] = R.get_average_image(start_frame=self.parameters['fluo_cam_frame_skip'],
                                                                      calc_min=True)

        return min_background

    def get_bg_mask_from_bg_images(self,bg_images,threshold=22,min_blob_size=700,max_hole_size=5000,dilation_radius=7):
        bg_masks = {}
        for c in ['red','yellow']:
            _, bg_mask = cv2.threshold(bg_images[c], threshold, 255, cv2.THRESH_BINARY)
            cleaned_bg_mask = morphology.remove_small_objects(bg_mask.astype(bool), min_size=min_blob_size)
            filled_bg_mask = morphology.remove_small_holes(cleaned_bg_mask,area_threshold=max_hole_size)
            bg_masks[c] = filled_bg_mask
        combined_bg_mask = bg_masks['red'] | bg_masks['yellow']
        dilated_bg_mask = morphology.binary_dilation(combined_bg_mask,np.ones([dilation_radius,dilation_radius]))
        return dilated_bg_mask

    # def define_background_to_remove_from_fluorescence_frame(self, trim_coordinates, frame_index, norm_mat=None):
    #    pass

    def read_missing_frames_file(self,tags=False):
        full_path = self.root_path + sep + 'missing_frames.csv'
        if tags:
            full_path = full_path[0:-4] + '_tags.csv'
        with open(full_path,'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            missing_frames = []
            for row in csv_reader:
                if len(row)>0:
                    missing_frames.append(int(row[0]))
        return missing_frames

    def get_transformation_parameters(self):
        """ Chooses an example frame from the end of the experiment
        Checks for saved transformation parameters
        If parameters exist, shows transformation for the example frame and requests user's approval
        If approved, returns saved parameters.
        If not approved, or if there are no saved parameters,
            creates parameters based on the example frame and saves them """

        # choose example frame from end of experiment
        example_frame = self.total_number_of_frames - 10
        # make sure the example frame is not a frame which has no matching fluorescence image
        while example_frame in self.missing_frames or example_frame in self.missing_frames_tags:
            example_frame += 1
            if example_frame == self.total_number_of_frames:
                example_frame = self.total_number_of_frames - 50

        transformation_parameters_exist = hf.check_if_file_exists(self.root_path,'transformation_parameters.pickle')
        if transformation_parameters_exist:
            with open(self.root_path + sep + 'transformation_parameters.pickle','rb') as f:
                trim_coordinates, frame_sizes, transformation_matrix = pickle.load(f)

            self.go_to_frame(example_frame)
            th, color, _ = self.set_binarize_parameters()

            _, tag_image = self.TagsReader.read_frame()
            tag_frame = Frame.TagsFrame(tag_image)
            tag_frame.full_transform(trim_coordinates, frame_sizes, transformation_matrix)

            _, fluo_image = self.FluoReader.read_frame()
            fluo_frame = Frame.FluorescenceFrame(fluo_image)
            fluo_frame.full_transform(trim_coordinates, th)

            output_frame = Frame.OutputFrame(tag_frame, fluo_frame)
            output_frame.overlay(color=color)

            plt.figure(1)
            output_frame.show()
            plt.pause(2)
            user_answer = input('Enter if everything is OK, input anything else to redo')

            if user_answer != '':
                trim_coordinates, frame_sizes = self.find_trim_coordinates(example_frame)
                example_frames = np.random.random_integers(100, self.total_number_of_frames, [1, 3])
                transformation_matrices = []
                for ex_frame1 in example_frames[0]:
                    # make sure the example frame is not a frame which has no matching fluorescence image
                    while ex_frame1 in self.missing_frames or ex_frame1 in self.missing_frames_tags:
                        ex_frame1 += 1
                        if ex_frame1 == self.total_number_of_frames:
                            ex_frame1 = self.total_number_of_frames - 50

                    transformation_matrix1 = self.get_transformation_matrix(ex_frame1, trim_coordinates, frame_sizes)
                    transformation_matrices.append(transformation_matrix1)
                transformation_matrix = np.mean(transformation_matrices, axis=0)
                with open(self.root_path + sep + 'transformation_parameters.pickle', 'wb') as f:
                    pickle.dump([trim_coordinates, frame_sizes, transformation_matrix], f)
        else:
            trim_coordinates, frame_sizes = self.find_trim_coordinates(example_frame)

            example_frames = np.random.random_integers(100,self.total_number_of_frames,[1,3])
            transformation_matrices = []
            for ex_frame1 in example_frames[0]:
                # make sure the example frame is not a frame which has no matching fluorescence image
                while ex_frame1 in self.missing_frames or ex_frame1 in self.missing_frames_tags:
                    ex_frame1 += 1
                    if ex_frame1 == self.total_number_of_frames:
                        ex_frame1 = self.total_number_of_frames - 50

                transformation_matrix1 = self.get_transformation_matrix(ex_frame1, trim_coordinates, frame_sizes)
                transformation_matrices.append(transformation_matrix1)
            transformation_matrix = np.mean(transformation_matrices, axis=0)
            with open(self.root_path + sep + 'transformation_parameters.pickle','wb') as f:
                pickle.dump([trim_coordinates, frame_sizes, transformation_matrix], f)

        return trim_coordinates, frame_sizes, transformation_matrix

    def find_trim_coordinates(self, frame_index):
        dict_keys = ['Fluorescence', 'Tags']
        trim_coordinates = dict(zip(dict_keys, [[],[]]))
        frame_sizes = dict(zip(dict_keys, [[], []]))
        self.go_to_frame(frame_index)
        for Reader_c, key in zip([self.FluoReader, self.TagsReader], dict_keys):
            _, image_c = Reader_c.read_frame()
            frame_c = Frame.Frame(image_c)
            trim_coordinates[key], frame_sizes[key] = frame_c.get_trim_coordinates()
        return trim_coordinates, frame_sizes

    def sync_cameras_go_to_start(self):
        self.FluoReader.frame_index = self.parameters['fluo_cam_frame_skip']
        self.TagsReader.frame_index = self.parameters['tag_cam_frame_skip']

    def go_to_frame(self, frame_index):
        if frame_index in self.missing_frames:
            print('given frame missing fluorescence image')
        self.TagsReader.frame_index = frame_index + self.parameters['tag_cam_frame_skip']
        skipped_frames_till_now = sum(i <= frame_index for i in self.missing_frames) - sum(i <= frame_index for i in self.missing_frames_tags)
        if frame_index in self.missing_frames_tags:
            self.FluoReader.frame_index = frame_index + self.parameters['fluo_cam_frame_skip'] - skipped_frames_till_now + 1
        else:
            self.FluoReader.frame_index = frame_index + self.parameters['fluo_cam_frame_skip'] - skipped_frames_till_now

    def get_transformation_matrix(self, example_frame, trim_coordinates, frame_sizes):

        self.go_to_frame(example_frame)
        _, tag_image = self.TagsReader.read_frame()
        _, fluo_image = self.FluoReader.read_frame()

        while True:

            tag_frame = Frame.TagsFrame(tag_image)
            tag_frame.frame = tag_frame.trim_frame(trim_coordinates['Tags'])
            tag_frame.flip()

            fluo_frame = Frame.FluorescenceFrame(fluo_image)
            fluo_frame.frame = fluo_frame.trim_frame(trim_coordinates['Fluorescence'])

            plt.figure(1)
            plt.imshow(tag_frame.frame)

            plt.figure(2)
            plt.imshow(fluo_frame.frame)
            plt.pause(2)

            print("click on 3 significant points in fluo image")
            ptsF = plt.ginput(3,timeout=-1)

            plt.figure(1)
            print("click on same 3 points in tags image")
            ptsT = plt.ginput(3,timeout=-1)

            transformation_matrix = cv2.getAffineTransform(np.float32(ptsT), np.float32(ptsF))

            plt.close('all')

            tag_frame.transform(transformation_matrix, (frame_sizes['Fluorescence'][1], frame_sizes['Fluorescence'][0]))
            fluo_frame.binarize(threshold=20)

            dual_frame = Frame.OutputFrame(tag_frame, fluo_frame)
            dual_frame.overlay()
            plt.figure(3)
            dual_frame.show()

            plt.pause(2)
            user_answer = input('Enter if everything is OK, input anything else to redo')
            plt.close(3)
            if user_answer is '':
                break
            else:
                tag_frame.frame = tag_frame.original_frame
                fluo_frame.frame = fluo_frame.original_frame

        return transformation_matrix

    def create_combined_frame(self,trim_coordinates,frame_sizes,transformation_matrix,bg_masks, norm_mats=None, frame_index=None,th=None,
                              acquisition=None,color=None):

        if frame_index is not None:
            self.go_to_frame(frame_index)

        if th is None:
            th, color, acquisition = self.set_binarize_parameters()

        if type(bg_masks) is dict:
            bg_mask = bg_masks[acquisition]
        elif type(bg_masks) is np.ndarray:
            bg_mask = bg_masks

        # read frames
        available_frame_to_read_tags, tag_image = self.TagsReader.read_frame()
        available_frame_to_read_fluo, fluo_image = self.FluoReader.read_frame()

        # transform frames
        tag_frame = Frame.TagsFrame(tag_image)
        tag_frame.full_transform(trim_coordinates, frame_sizes, transformation_matrix)
        fluo_frame = Frame.FluorescenceFrame(fluo_image)
        if norm_mats is None:
            fluo_frame.full_transform(trim_coordinates, th, bg_mask)
        else:
            fluo_frame.full_transform(trim_coordinates, th, bg_mask, norm_mat= norm_mats[acquisition])

        # combine frames
        output_frame = Frame.OutputFrame(tag_frame, fluo_frame)
        output_frame.overlay(color=color)

        available_frame_to_read = available_frame_to_read_fluo and available_frame_to_read_tags

        return tag_frame, fluo_frame, output_frame, available_frame_to_read, acquisition, th


class Experiment2Colors(Experiment):
    def __init__(self,path,fluoReader_type = 'img',get_timestamps=True,tags_vid_prefix='GT'):
        super().__init__(path,fluoReader_type,get_timestamps=get_timestamps,tags_vid_prefix=tags_vid_prefix)
        self.FluorescencePath = self.path + sep + 'UIfreezer'
        self.TagsPath = self.path + sep + 'Bugtag output'
        # self.FluoReader = Reader.ImageReader(self.FluorescencePath)
        # self.TagsReader = Reader.VideoReader(self.TagsPath)
        self.yellow_threshold, self.red_threshold, self.fluo_cam_frame_skip, self.tag_cam_frame_skip = self.read_parameters_file()

    def read_parameters_file(self):
        full_path = self.root_path + sep + 'Parameters.csv'
        parameters = pd.read_csv(full_path)
        yellow_threshold = int(parameters.Yellow_threshold[0])
        red_threshold = int(parameters.Red_threshold[0])
        fluo_cam_frame_skip = int(parameters.fluo_cam_frame_skip)
        tag_cam_frame_skip = int(parameters.tag_cam_frame_skip)
        return yellow_threshold, red_threshold, fluo_cam_frame_skip, tag_cam_frame_skip

    def define_background_to_remove_from_fluorescence_frame(self,trim_coordinates,frame_index, norm_mat=None):
        dict_keys = ['BLGF', 'GLRF']
        bg_masks = dict(zip(dict_keys, [None, None]))
        self.go_to_frame(frame_index)
        for f in [1,2]:
            plt.figure(f)
            bg_mask, acquisition = super().define_fluorescence_background_by_threshold(trim_coordinates, norm_mat)
            bg_masks[acquisition] = bg_mask
        return bg_masks

    def get_food_source_masks(self,trim_coordinates,frame_sizes,transformation_matrix):
        keys = ['red','yellow']
        food_source_masks=dict(zip(keys, [None, None]))
        for key in keys:
            raw_mask = cv2.imread(self.path + sep + 'food_source_mask_' + key + '.tif',cv2.IMREAD_GRAYSCALE)
            mask_frame = Frame.TagsFrame(raw_mask)
            mask_frame.full_transform(trim_coordinates,frame_sizes,transformation_matrix)
            food_source_masks[key] = mask_frame.frame
        return food_source_masks


class ExperimentOneColor(Experiment):
    def __init__(self,path):
        super().__init__(path)

    def define_background_to_remove_from_fluorescence_frame(self,trim_coordinates,frame_index, norm_mat=None):
        self.go_to_frame(frame_index)
        bg_mask, acquisition = super().define_fluorescence_background_by_threshold(trim_coordinates, norm_mat)
        bg_masks = {acquisition: bg_mask}
        return bg_masks
