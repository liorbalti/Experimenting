import numpy as np
import pandas as pd
import csv
import os
import helper_functions as hf

sep = os.sep


class AntsInfo:
    def __init__(self, root_path, tagfile_format='python'):
        self.root_path = root_path
        self.path = root_path + sep + 'Bugtag output'
        self.file_list = self.list_bugtag_files()
        self.tag_list = self.get_tag_list(self.file_list[0], tagfile_format)
        self.angle_corrections = self.get_angle_corrections()
        self.ant_parameters_list = self.make_ant_parameters_list()

    def list_bugtag_files(self):
        file_list = []
        for _, _, files in os.walk(self.path):
            file_list.extend([f for f in files if f.endswith('csv') and f.startswith('GT')])
        file_list.sort()
        return file_list

    def get_tag_list(self, filename, tagfile_format):
        with open(self.path + sep + filename, 'r') as file:
            headers = file.readline()
            split_headers = headers.split(',')
        if tagfile_format == 'python':
            taglist = [x[0:-2] for x in split_headers[slice(2, len(split_headers), 4)]]
        elif tagfile_format == 'bugtag':
            taglist = [x[2:] for x in split_headers[slice(1, len(split_headers), 4)]]
        else:
            print("invalid bugtag format " + tagfile_format + ". Should be 'bugtag' or 'python'.")
        return taglist

    def read_tag_data_into_panda(self):
        tags_df = pd.DataFrame()
        for filename in self.file_list:
            p = pd.read_csv(self.path + sep + filename)
            tags_df = pd.concat([tags_df, p], ignore_index=True)
            print(tags_df.shape)

    def parse_tag_data(self, tagfile_format='python'):
        ants_info_all_frames = []
        for filename in self.file_list:
            reader = csv.DictReader(open(self.path + sep + filename, 'r'))
            for frame_dict in reader:
                ants_frame_info = AntObjectsList()
                for tag in self.tag_list:
                    tag_identified = self.is_tag_identified(tag, frame_dict, tagfile_format)
                    if tag_identified:
                        if tagfile_format == 'bugtag':
                            ant = CurrentAntData(tag, int(float(frame_dict['Error-' + tag])),
                                                 float(frame_dict['X-' + tag]), float(frame_dict['Y-' + tag]),
                                                 float(frame_dict['Angle-' + tag]))
                        elif tagfile_format == 'python':
                            ant = CurrentAntData(tag, int(float(frame_dict[tag + '-error'])),
                                                 float(frame_dict[tag + '-x']),
                                                 float(frame_dict[tag + '-y']), float(frame_dict[tag + '-angle']))
                        else:
                            print("invalid tagfile format. shoud be 'python' or 'bugtag'.")
                        ants_frame_info.append(ant)
                ants_info_all_frames.append(ants_frame_info)
        return ants_info_all_frames

    def is_tag_identified(self, tag, frame_dict, tagfile_format):
        if tagfile_format == 'python':
            tag_identified = float(frame_dict[tag + '-x']) > -1
        elif tagfile_format == 'bugtag':
            tag_identified = frame_dict['X-' + tag] is not ''
        else:
            print('invalid tagfile format. should be "bugtag" or "python"')
        return tag_identified

    def get_angle_corrections(self):
        reader = csv.DictReader(open(self.root_path + sep + 'angle_correction.csv'))
        angle_corrections = next(reader)
        return angle_corrections

    def make_ant_parameters_list(self):
        ant_parameters_list = AntObjectsList()
        for tag in self.tag_list:
            ant_parameters = AntParameters(tag, self.angle_corrections[tag])
            ant_parameters_list.append(ant_parameters)
        return ant_parameters_list


class AntObjectsList(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            current_id_list = [x.id for x in self]
            new_key = [x[0] for x in enumerate(current_id_list) if x[1] == key]
            return super().__getitem__(new_key[0])
        else:
            return super().__getitem__(key)

    def remove_out_of_bounds_detections(self):
        for idx, ant in enumerate(self):
            if ant.x is None:
                del self[idx]


class CurrentAntData:
    def __init__(self, id, error, original_x, original_y, original_angle):
        self.x = None
        self.y = None
        self.angle = None
        self.error = error
        self.id = id
        self.crop_intensity = None
        self.crop_area = None
        self.food_color = None
        self.crop_x = None
        self.crop_y = None
        self.original_x = original_x
        self.original_y = original_y
        self.original_angle = original_angle
        self.angle_correction = None

    def to_dict(self, keys):
        raw_dict = self.__dict__
        id_dict = {'a' + self.id + '-' + key: value for key, value in raw_dict.items() if key in keys}
        return id_dict

    def transform_coordinates(self, trim_coordinates, frame_sizes, transformation_matrix):
        # Trim:
        trimmed_x = self.original_x - trim_coordinates['Tags'][0]
        trimmed_y = self.original_y - trim_coordinates['Tags'][2]
        # Flip:
        flipped_x = frame_sizes['Tags'][1] - trimmed_x
        flipped_y = trimmed_y
        flipped_angle = -self.original_angle % 360
        # Transform:
        transformed_x, transformed_y = np.dot(transformation_matrix, [[flipped_x], [flipped_y], [1]])
        # Angle correction:
        corrected_angle = flipped_angle - float(self.angle_correction)  # subtraction because of flipping

        self.x = transformed_x[0]
        self.y = transformed_y[0]
        self.angle = corrected_angle - 180 % 360

    def check_out_of_bounds(self, frame_sizes):
        out_of_bounds = self.x < 0 or self.y < 0 or self.x > frame_sizes['Fluorescence'][1] or self.y > \
                        frame_sizes['Fluorescence'][0]
        return out_of_bounds

    def make_rectangle(self, ant_parameters):
        w = ant_parameters.rectangle_size[0]  # rectangle width
        h = ant_parameters.rectangle_size[1]  # rectangle height
        d = ant_parameters.dist_from_tag  # distance from tag to rectangle
        rect = hf.make_rectangle(self.x, self.y, self.angle, w, h, d)
        return rect


class AntParameters:
    def __init__(self, id, angle_correction, rect_size=(30, 130), dist_from_tag=50):
        self.id = id
        self.angle_correction = angle_correction
        self.rectangle_size = rect_size
        self.dist_from_tag = dist_from_tag
