""" For Reading frames and videos from the 2-color setup. """

import cv2
import os
import helper_functions as hf
from datetime import datetime as dt
import numpy as np
sep = os.sep


class FileReader:

    """ Super class for readers of different file types.
    Lists files and subfolders.
    """

    def __init__(self, path, extension, prefix=None):
        self.root_path = path
        self.extension = extension
        self.file_list, self.subfolder_list = self.list_files_and_subfolders(extension, prefix)
        self._frame_index = None

    @property
    def frame_index(self):
        return self._frame_index

    def list_files_and_subfolders(self, file_extension, prefix=None):
        file_list=[]
        subfolder_list=[]
        if prefix is None:
            for folder_path,subfolders,files in os.walk(self.root_path):
                file_list.extend([folder_path + sep + f for f in files if f.endswith(file_extension)])
                subfolder_list.extend(subfolders)
        else:
            for folder_path,subfolders,files in os.walk(self.root_path):
                file_list.extend([folder_path + sep + f for f in files if (f.endswith(file_extension)
                                                                           and f.startswith(prefix))])
                subfolder_list.extend(subfolders)
        file_list.sort()
        subfolder_list.sort()
        return file_list, subfolder_list

    def read_frame(self, index=None):
        pass

    def get_timestamps(self):
        pass

    def get_average_image(self, n_images=60, random=True, start_frame=0, end_frame=None, calc_min=False):
        if end_frame is None:
            end_frame = len(self.file_list) - 2

        # if insufficient frame range, calculate on all images in range:
        frame_range = end_frame - start_frame
        has_enough = frame_range >= n_images
        if not has_enough:
            print(f'Limited by frame range {start_frame} to {end_frame}. Calculating average for {frame_range} images.')
            n_images = frame_range
            random = False

        # generate list of frames to read
        if random:
            frames_to_read = np.random.choice(np.arange(start_frame, end_frame), n_images)
        else:
            frames_to_read = np.arange(start_frame, start_frame + n_images)

        if not calc_min:
            for ii, f in enumerate(frames_to_read):
                _, I = self.read_frame(f)
                if ii == 0:
                    sumI = np.zeros(I.shape)
                sumI += I

            meanI = sumI / n_images
            return meanI
        else:
            for ii, f in enumerate(frames_to_read):
                _, I = self.read_frame(f)
                if ii == 0:
                    minI = 255 * np.ones_like(I)
                minI = cv2.min(I, minI)
            return minI


class ImageReader(FileReader):
    def __init__(self,path,extension='tif', prefix='UI', frame_index=0):
        super().__init__(path, extension, prefix)
        self.check_out_of_bounds(frame_index)
        self._frame_index = frame_index
        self.next_frame_to_read = self.file_list[self._frame_index]
        self.acquisition = self.get_acquisition_from_filename()

    def get_acquisition_from_filename(self):
        filename_acquisition = self.next_frame_to_read[-8:-4]
        # switch to opposite acquisition because it is wrong in the filenames:
        if filename_acquisition in ('BLGF', '10ms'):
            acquisition = 'GLRF'
        elif filename_acquisition in ('GLRF', '33ms', '30ms'):
            acquisition = 'BLGF'
        else:
            acquisition = None
            print('Invalid acquisition ' + filename_acquisition)
        return acquisition

    def get_timestamps(self):
        timestamps = []
        for file in self.file_list:
            split_filename = file.split('_')
            filename = split_filename[-2]+ '_' + split_filename[-1]
            split_again = filename.split('exp')
            timestamp = split_again[0]
            date_time = dt.strptime(timestamp, '%d-%m-%y_%H.%M.%S')
            timestamps.append(hf.datenum(date_time))
        return timestamps

    @FileReader.frame_index.setter
    def frame_index(self, value=0):
        self.check_out_of_bounds(value)
        self._frame_index = value
        self.next_frame_to_read = self.file_list[self._frame_index]

    def check_out_of_bounds(self,index):
        try:
            dummy = self.file_list[index]
            out_of_bounds = False
        except IndexError as err:
            print('Index provided is out of bounds.')
            out_of_bounds = True
        return out_of_bounds

    def read_frame(self,index=None, grayscale=True):
        if index is not None:
            self.check_out_of_bounds(index)
            self.next_frame_to_read = self.file_list[index]
            self.acquisition = self.get_acquisition_from_filename()

        image_object = Image(self.next_frame_to_read, grayscale)
        image = image_object.image

        self.frame_index = self.frame_index + 1
        try:
            self.next_frame_to_read = self.file_list[self.frame_index]
            self.acquisition = self.get_acquisition_from_filename()
        except IndexError:
            self.frame_index = self.frame_index - 1
            print('Reached last frame.')
            return False, image

        return True, image


class VideoReader(FileReader):
    # region Constructor
    def __init__(self, path, extension='avi', prefix = None, video_index=0, frame_in_video_index=0, frame_index=0, get_timestamps = False):
        super().__init__(path, extension, prefix)
        video_lengths = self.read_video_lengths_file()
        self.video_lengths = [int(float(v[1])) for v in video_lengths]
        self.cumulative_video_lengths = hf.cumulative_list(self.video_lengths)
        if get_timestamps is True:
            self.timestamps = self.get_timestamps()

        if (video_index is not 0 or frame_in_video_index is not 0) and frame_index is not 0:
            raise IndexError('Cannot provide frame index, video index and frame_in_video index simultaneously.')
        elif video_index is not 0 or frame_in_video_index is not 0:
            self.video_and_frame_in_video_check_out_of_bounds(video_index,frame_in_video_index)
            self._video_index = video_index
            self._frame_in_video_index = frame_in_video_index
            self.convert_video_and_frame_in_video_index_into_frame_index()
        elif frame_index is not 0:
            self.frame_check_out_of_bounds(frame_index)
            self._frame_index = frame_index
            self.convert_frame_index_into_video_and_frame_in_video_index()
        else:
            self._video_index = video_index
            self._frame_in_video_index = frame_in_video_index
            self._frame_index = frame_index

        self.current_video_object = Video(self.file_list[self._video_index])
        self.current_video_object.set_video_to_frame(self._frame_in_video_index)
    # endregion

    # region Public Properties
    @FileReader.frame_index.setter
    def frame_index(self, value=0):
        self.frame_check_out_of_bounds(value)
        self._frame_index = value
        self.convert_frame_index_into_video_and_frame_in_video_index()
        self.current_video_object = Video(self.file_list[self._video_index])
        self.current_video_object.set_video_to_frame(self._frame_in_video_index)

    @property
    def video_index(self):
        return self._video_index

    @video_index.setter
    def video_index(self,value):
        print('Cannot set video_index separately. Use set_video_and_frame_in_video_indices.')

    @property
    def frame_in_video_index(self):
        return self._frame_in_video_index

    @frame_in_video_index.setter
    def frame_in_video_index(self, value):
        print('Cannot set frame_in_video_index separately. Use set_video_and_frame_in_video_indices.')

    def set_video_and_frame_in_video_indices(self, video_index, frame_in_video_index):
        self.video_and_frame_in_video_check_out_of_bounds(video_index,frame_in_video_index)
        self._video_index = video_index
        self._frame_in_video_index = frame_in_video_index
        self.convert_video_and_frame_in_video_index_into_frame_index()
        self.current_video_object = Video(self.file_list[self._video_index])
        self.current_video_object.set_video_to_frame(self._frame_in_video_index)
    # endregion

    # region Video Lengths File Read/Create
    def read_video_lengths_file(self):
        full_path = self.root_path + sep + 'video_lengths.csv'
        csvdata = hf.read_from_csv(full_path)
        if csvdata is None:
            print('Could not find video lengths file. Creating!')
            csvdata = self.create_video_lengths_file()
        else:
            if csvdata.__len__() is not self.file_list.__len__():
                raise Exception('\nFile length detected: ' + str(csvdata.__len__()) +
                                '\nExpected file length: ' + str(self.file_list.__len__()))
        return csvdata

    def create_video_lengths_file(self):
        full_path = self.root_path + sep + 'video_lengths.csv'
        csvdata = []
        for file in self.file_list:
            vid_object = Video(file)
            number_of_frames = vid_object.number_of_frames
            new_row = [file, number_of_frames]
            csvdata.append(new_row)
        hf.write_to_csv(full_path,csvdata)
        return csvdata
    #endregion

    # region Converters
    def convert_frame_index_into_video_and_frame_in_video_index(self):
        self._video_index = next(x[0] for x in enumerate(self.cumulative_video_lengths) if self._frame_index < x[1])-1
        self._frame_in_video_index = self._frame_index - self.cumulative_video_lengths[self._video_index]

    def convert_video_and_frame_in_video_index_into_frame_index(self):
        self._frame_index = self.cumulative_video_lengths[self._video_index] + self._frame_in_video_index
    # endregion

    #  region Input Validation Methods
    def frame_check_out_of_bounds(self, index):
        if index > self.cumulative_video_lengths[-1] - 1:
            raise IndexError('Frame index provided is out of bounds (too large).')

    def video_and_frame_in_video_check_out_of_bounds(self, video_index, frame_in_video_index):
        if video_index >= len(self.video_lengths):
            raise IndexError('Video index provided is out of bounds (too large).')
        if frame_in_video_index >= self.video_lengths[frame_in_video_index]:
            raise IndexError('Frame in video index provided is out of bounds (too large).')
    # endregion

    # region Methods

    def get_timestamps(self):
        movie_timing_files, _ = self.list_files_and_subfolders('txt')
        timestamps = []
        for file in movie_timing_files:
            with open(file) as f_input:
                for line in f_input.readlines():

                    # read timestamp depending on timestamps file format
                    if '\t' in line:
                        split_line = line.split('\t')
                        timestamp = float(split_line[0])
                    elif line.startswith('GT'):
                        gg = line.split('_')
                        date = gg[1]
                        gg2 = gg[2].split('exp')
                        time = gg2[0]
                        timestring = date + '_' + time
                        time_obj = dt.strptime(timestring, '%d-%m-%y_%H.%M.%S.%f')
                        timestamp = dt.timestamp(time_obj)
                    else:
                        timestamp = float(line)

                    if timestamps:
                        timestamps.append(timestamp-timestamps[0])
                    else:
                        timestamps.append(timestamp)
        timestamps[0] = 0.0
        return timestamps

    def read_frame(self, index=None):
        if index is not None:
            self.frame_check_out_of_bounds(index)
            self._frame_index = index
            self.convert_frame_index_into_video_and_frame_in_video_index()
            self.current_video_object = Video(self.file_list[self._video_index])
            self.current_video_object.set_video_to_frame(self._frame_in_video_index)

        # reading frame corresponding to frame_index.
        image = self.current_video_object.read()

        # checking if we did not reach the last frame overall.
        if self._frame_index == self.cumulative_video_lengths[-1]-1:
            # if so, print to console and return false to indicate no more frames available (outer loop should be terminated).
            print('Reached last frame.')
            return False, image

        # checking if we now read the last frame of the current video.
        if self._frame_in_video_index == self.video_lengths[self._video_index]-1:
            # if so, change indices accordingly and return.
            self._video_index +=1
            self._frame_in_video_index = 0
            self._frame_index += 1
            self.current_video_object = Video(self.file_list[self._video_index])
            self.current_video_object.set_video_to_frame(self._frame_in_video_index)
            return True, image

        # if we are not at the end of any video, advance indices normally.
        self._frame_index += 1
        self._frame_in_video_index += 1
        return True, image
    #endregion


class Video:
    def __init__(self, path):
        # frame count always starts at 0!!!
        self.path = path
        self.current_frame = 0
        self.captured_video = Video.capture(self.path)
        self.number_of_frames  = self.captured_video.get(cv2.CAP_PROP_FRAME_COUNT)

    @staticmethod
    def capture(path):
        captured_video = cv2.VideoCapture(path)
        return captured_video

    def set_video_to_frame(self, frame):
        _ = self.captured_video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        self.current_frame = frame

    def read(self):
        _, frame = self.captured_video.read()
        self.current_frame = self.current_frame+1
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame


class Image:
    def __init__(self, path, is_grayscale):
        self.path = path
        self.is_grayscale = is_grayscale
        self.image = self.read_image()

    def read_image(self):
        if self.is_grayscale:
            image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(self.path)
        return image


class MissingFramesSolver:
    def __init__(self,current_frame,missing_frames,tag_reader,fluo_reader):
        self.current_frame = current_frame
        self.missing_frames = missing_frames

    def deal_with_missing_fluo_frame(self):
        pass

    def deal_with_missing_tag_frame(self):
        pass