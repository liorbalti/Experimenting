"""
This script interactively gets correction angles for ant tags.

BACKGROUND ON OUR DATA:
-----------------------
A typical experiment consists of several consecutive videos of an ant colony where each ant is tagged with a unique
barcode (the tag is attached to the "back" of the ant). The videos are analyzed with the Bugtag software, which detects
and recognizes the tags, outputing a csv file for each video. Each row of the csv file corresponds to a single frame in
the video. The columns of the csv file are:
'frame #' - the index of the frame
'recognitions' - the total number of tags recognized in that frame
and for each tag ID 'X'
'X-x' - the x-pixel coordinate of tag X in that frame
'X-y' - the y-pixel coordinate of tag X in that frame
'X-angle' - the orientation of tag X in that frame (in degrees)
'X-error' - an integer between 0 and 5 indicating how many 1-bit corrections were needed for tag recognition. Can be
    interpreted as the inverse of recognition confidence (0: highest confidence, 5: lowest confidence).
If tag 'X' is not recognized in that frame, the values in all of its 4 columns will be '-1'.

Since tags are attached to ants at random orientations, the orientation of the tag is not necessarily the heading
direction of the ant. The goal of this program is to obtain the correction angle of each tag, that is, the angular
difference between the tag orientation and the ant's heading direction. These correction angles will be used in further
analyses of the experiment.


INPUT:
------
experiment_path : str
    The full path of one folder that contains all experiment video files and their corresponding Bugtag output files.
output_path : str
    The full location where the output csv file will be saved

OUTPUT:
-------
The program writes a csv file containing the correction angle of each tag.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import cv2
import os
import random
import math
import scipy.stats as stats
import csv
sep = os.sep
import tkinter as tk
import tkinter.filedialog


# region: GENERAL FUNCTIONS

def calc_angle_diff(alpha, beta):
    """
    Calculates the angular difference between two angles alpha and beta (floats, in degrees).
    returns d_alpha: (float, in degrees between 0 and 360)
    """
    d = alpha - beta
    d_alpha = (d + 180) % 360 - 180
    return d_alpha


def insert_nan_row_in_beginning_of_df(df):
    top_row = pd.DataFrame({col: np.nan for col in df.columns},index=[0])
    df2 = pd.concat([top_row, df]).reset_index(drop=True)
    return df2

# endregion: GENERAL FUNCTIONS


# region: PATH SELECTION WINDOW

def path_selection_window():
    """
    Creates a tkinter window for selecting input and output paths.
    Returns experiment_path and output_path as strings.
    """

    # functions for button events
    def get_experiment_folder():
        path = tk.filedialog.askdirectory()
        experiment_path_var.set(path)

    def get_output_folder():
        path = tk.filedialog.askdirectory()
        output_path_var.set(path)

    # creating the tk window
    window = tk.Tk()
    window.title("Angle correction")

    experiment_path_var = tk.StringVar()
    output_path_var = tk.StringVar()

    tk.Label(window,text='Please select the following paths').grid(row=0,column=2,padx=20,pady=20)
    tk.Label(window,text='Experiment path:').grid(row=1,column=1,padx=20)
    tk.Label(window,text='Output path:').grid(row=2,column=1,padx=20,pady=20)

    e1 = tk.Entry(window, textvariable=experiment_path_var, width=80).grid(row=1, column=2)
    e2 = tk.Entry(window, textvariable=output_path_var, width=80).grid(row=2, column=2, pady=20)

    b1 = tk.Button(window,text='Select experiment folder', command=get_experiment_folder)
    b2 = tk.Button(window,text='Select output folder', command=get_output_folder)
    b1.grid(row=1,column=3,padx=20)
    b2.grid(row=2,column=3,padx=20,pady=20)

    b3 = tk.Button(window,text='OK',command=window.destroy).grid(row=3,column=2,pady=20)

    window.mainloop()

    return experiment_path_var.get(), output_path_var.get()

# endregion: PATH SELECTION WINDOW


# region: CLASSES

class TagFileReader:
    """
    A class for reading tag data of the files in the given folder path
    ...

    Attributes
    ----------
    - path : (str) the path of the folder of the experiment videos and corresponding Bugtag files
    - tag_file_list: (list) list of all Bugtag file names in the folder (as strings)
    - movie_file_list: (list) list of all video file names in the folder (as strings)
    - tags_df: (DataFrame) a single pandas DataFrame containing all Bugtag output data from all files
    - movie_lengths: (list) a list containing the number of frames in each video (as integers)
    - tag_list: (list) a list of all tag IDs (as strings)

    """
    def __init__(self, path, tagfile_format='python'):
        self.path = path
        if tagfile_format == 'python':
            self.tag_file_list = self.list_files(prefix='GT6600', extension='csv')
            self.movie_file_list = self.list_files(prefix='Tagged', extension='avi')
        elif tagfile_format == 'bugtag':
            self.tag_file_list = self.list_files(prefix='', extension='_tags.csv')
            self.movie_file_list = self.list_files(prefix='', extension='avi')
        else:
            print("invalid bugtag format " + tagfile_format + ". Should be 'bugtag' or 'python'.")
        self.tags_df, self.movie_lengths = self.read_tag_data_into_panda()
        self.tag_list = self.get_tag_list(tagfile_format)

    def list_files(self, prefix, extension):
        file_list = []
        for _, _, files in os.walk(self.path):
            file_list.extend([f for f in files if (f.endswith(extension) and f.startswith(prefix))])
        file_list.sort()
        return file_list

    def read_tag_data_into_panda(self):
        tags_df = pd.DataFrame()
        video_lengths = []
        for filename in self.tag_file_list:
            p = pd.read_csv(self.path + sep + filename)
            tags_df = pd.concat([tags_df, p], ignore_index=True, sort=False)
            video_lengths.append(len(p))
        video_lengths.insert(0, 0)  # inserting 0 as the first element in video_lengths. This is helpful for setting a
        # frame to read using a cumulative sum of the video_lengths list. (See MovieFrame.read_frame_from_movie)
        return tags_df, video_lengths

    def get_tag_list(self,tagfile_format):
        with open(self.path + sep + self.tag_file_list[0], 'r') as file:
            headers = file.readline()
        split_headers = headers.split(',')
        if tagfile_format == 'python':
            tag_list = [x[0:-2] for x in split_headers[slice(2, len(split_headers), 4)]]  # going over every 4th header
            # starting from the third gives all the x-coordinate headers, which have the format 'X-x' for each tag 'X'.
            # Indexing this from 0:-2 gives the tag ID 'X'.
        elif tagfile_format == 'bugtag':
            tag_list = [x[2:] for x in split_headers[slice(1, len(split_headers), 4)]]  # going over every 4th header
            # starting from the second gives all the X-coordinate headers, which have the format 'X-x' for each tag 'x'.
            # Indexing this from 2: gives the tag ID 'x'.
        else:
            print("invalid bugtag format " + tagfile_format + ". Should be 'bugtag' or 'python'.")
        return tag_list


class MovieFrame:
    """
    A class for reading and manipulating a video frame, focusing on a specified tag.
    ...

    Attributes
    -----------
    - reader: (TagFileReader) the TagFileReader object of the experiment's path
    - frame_index: (int) the general index of the frame (across all videos), between 0 and the total number of frames in
        the experiment.
    - tag: (Tag) a tag object of the specified tag
    - tag_x: (int) x-pixel coordinate of the specified tag in the frame
    - tag_y: (int) y-pixel coordinate of the specified tag in the frame
    - tag_angle: (int) orientation of the specified tag in the frame
    - frame: (ndarray) grayscale image of the frame from the video
    - cropped_image: (ndarray) a 400x400 grayscale image cropped from the frame around the tag coordinates. (If the tag
        is close to the edge of the frame, cropped_image may be smaller than 400x400).
    """

    def __init__(self, tag_file_reader, frame_index, tag):
        self.reader = tag_file_reader
        self.frame_index = frame_index
        self.tag = tag
        self.tag_x = int(self.tag.data.iloc[self.frame_index]['x'])
        self.tag_y = int(self.tag.data.iloc[self.frame_index]['y'])
        self.tag_angle = int(self.tag.data.iloc[self.frame_index]['angle'])
        self.frame = self.read_frame_from_movie()
        self.cropped_image = self.crop_around_tag()

    def read_frame_from_movie(self):
        cumulative_movie_lengths = np.cumsum(np.array(self.reader.movie_lengths))
        video_index = next(x[0] for x in enumerate(cumulative_movie_lengths) if self.frame_index < x[1]) - 1  # the
        # video index is the last video where the cumulative number of frames is smaller than the general frame index.
        frame_in_video_index = self.frame_index - cumulative_movie_lengths[video_index]
        captured_video = cv2.VideoCapture(self.reader.path + sep + self.reader.movie_file_list[video_index])
        _ = captured_video.set(cv2.CAP_PROP_POS_FRAMES, frame_in_video_index)
        _, frame = captured_video.read()
        if len(frame.shape) == 3:  # if frame is RGB convert to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame

    def crop_around_tag(self, s=200):  # s: half of width and length of output cropped image
        ny, nx = self.frame.shape
        around_tag = copy(self.frame[max(0, self.tag_y - s):min(self.tag_y + s, ny), max(0, self.tag_x - s):min(self.tag_x + s, nx)])
        # if tag is too close to frame edge, the output image will be smaller
        return around_tag

    def show(self, prompt_user=True):
        plt.ion()
        plt.figure(1)
        plt.imshow(self.cropped_image)
        if prompt_user:
            #plt.title(f"Ant {self.tag.id}\n Click on edge of ant's head. If bad image - press 'Esc' to get another")
            plt.title(f"Ant {self.tag.id}\n Click on ant's crop. If bad image - press 'Esc' to get another")

    def draw_arrow(self, r=100, s=200):
        """
        draws a red arrow of length 'r' from the ant's tag toward her head (using the obtained correction angle).
        's' is the same as in MovieFrame.crop_around_tag, needed for locating the tag in the cropped image.
        """
        angle = np.deg2rad(self.tag_angle + 180 + self.tag.correction_angle)
        plt.arrow(min(s, self.tag_x), min(s, self.tag_y), r * math.cos(angle), r * math.sin(angle), color=[1, 0, 0])


class Tag:
    """
    A class that represents a tag.
    ...

    Attributes
    ----------
    -id: (str) ID of the tag
    -data: (DataFrame)  a pandas DataFrame containing this tags data. Each row is a frame in the experiment. Columns are
        'x','y','angle' and 'error' (see data description at top of script for details)
    -random_frames: ('list_iterator') an iterator over shuffled frame indexes where the tag was recognized with 0 error.
        If the number of recognitions with 0 error is less than a minimum defined by 'min_detections', the tag is
        considered as a false recognition and 'random_frames' will be None.
    -correction_angle: ('float') the angular difference between the tag orientation and the ant's heading direction (in
        degrees). Initiated as 0, and updated as information is obtained from user.
    -user_approved: a flag indicating whether the tag's correction_angle is approved by the user. As long as it is false
        the program will keep asking the user for input on this tag before moving to the next tag.
    """

    def __init__(self, tag_id, tags_df, min_detections=1000, bugtag_format='bugtag'):
        self.id = tag_id
        self.data = self.get_tag_data(tags_df,bugtag_format)
        self.random_frames = self.permute_frames(min_detections)
        self.correction_angle = 0
        self.user_approved = False

    def get_tag_data(self, tags_df, bugtag_format):
        if bugtag_format =='python':
            tag_data = tags_df.filter(regex='^'+str(self.id)+'-')  # get tag's columns from the big tags DataFrame
            column_name_idx = 1
        elif bugtag_format == 'bugtag':
            tag_data = tags_df.filter(regex='-'+str(self.id)+'$')
            column_name_idx = 0
            tag_data = insert_nan_row_in_beginning_of_df(tag_data)
        else:
            print("invalid bugtag format " + bugtag_format + ". Should be 'bugtag' or 'python'.")
        original_columns = tag_data.columns
        split_columns = [x.split('-') for x in original_columns]
        new_columns = [x[column_name_idx].lower() for x in split_columns]  # remove tag ID from the columns' names
        tag_data.columns = new_columns
        return tag_data

    def permute_frames(self, min_detections):
        high_certainty_frames = self.data[(self.data['error'] == 0)].index.to_list()  # list of indexes of frames with
        # 0 error recognition
        random.shuffle(high_certainty_frames)
        if len(high_certainty_frames) > min_detections:  # if tag was recognized with 0 error for enough frames
            return iter(high_certainty_frames)
        else:  # if tag was not recognized with 0 error for enough frames it is considered as a false recognition.
            return None

    def tag_loop(self, tag_file_reader, n=3):
        """
        An interactive loop to get the tag's correction_angle.

        This loop displays random images of the ant with the tag, prompting the user to click on the ant's head to get
        her heading direction. If the user decides that the displayed frame is bad (e.g. the ant is standing in a
        posture that makes it difficult to define her heading direction), he/she may press 'Esc' to get a new random
        image. The difference between the user's angle and the tag's angle is recorded until there are 'n' such angle
        differences. The tag's correction_angle is the circular mean of the n angle differences.

        At the end of the process, several random images of the ant with the tag are displayed with an arrow drawn from
        the tag to the direction where her head should be (considering the tag's correction angle). The user is prompted
        to approve the correction angle (for exiting the loop), or disapprove to start the process over.
        """
        while not int(self.user_approved):
            angle_diffs = []
            while len(angle_diffs) < n:
                # display next random frame:
                frame_index = next(self.random_frames)
                mf = MovieFrame(tag_file_reader, frame_index, tag=self)
                mf.show()
                ant_heading_direction = self.get_user_angle(frame_index)
                if ant_heading_direction:  # false means that user pressed 'Esc'
                    d_alpha = calc_angle_diff(ant_heading_direction, mf.tag_angle + 180)
                    angle_diffs.append(d_alpha)
                    print("angle_diffs", angle_diffs)
            self.correction_angle = np.rad2deg(stats.circmean(np.deg2rad(angle_diffs)))  # circular mean of angle diffs
            plt.close(1)
            self.get_user_approval(tag_file_reader)
            if self.user_approved == 'x':  # 'x' means the correction is approved and the program will stop (will not
                # continue to a new tag, see 'main' function).
                break

    def get_user_angle(self, frame_index, s=200):
        x_tag = min(s, self.data.iloc[frame_index]['x'])  # location of tag in cropped frame
        y_tag = min(s, self.data.iloc[frame_index]['y'])
        p = np.array(plt.ginput(1, timeout=0))  # user's point coordinates
        if len(p) > 0:  # false means user pressed 'Esc'
            alpha = np.rad2deg(np.arctan2(p[0][1] - y_tag, p[0][0] - x_tag))  # angle from tag to user's point
            return np.round(alpha, 1)
        else:
            return None

    def get_user_approval(self, tag_file_reader, n=6):
        """
        Displays n random images of the tag with an arrow drawn from the tag to the ant's head (based on tag's
        correction_angle) for user's approval
        """
        plt.ion()
        fig, ax = plt.subplots(2, int(np.ceil(n/2)), num=2)
        # plt.get_current_fig_manager().window.showMaximized()
        plt.get_current_fig_manager().full_screen_toggle()
        for m in range(1, n+1):
            f = next(self.random_frames)
            mf = MovieFrame(tag_file_reader, f, tag=self)
            plt.subplot(2, int(np.ceil(n/2)), m)
            plt.axis('off')
            plt.imshow(mf.cropped_image)
            mf.draw_arrow()
        plt.suptitle(f"Ant {self.id}: angle ok [1]? or redo this ant [0]? \n \n Press ['x'] to approve, save and exit")
        fig.canvas.mpl_connect('key_press_event', self.press)
        plt.waitforbuttonpress(timeout=-1)
        user_approved = self.user_approved
        plt.close(2)
        return user_approved

    def press(self, event):
        self.user_approved = event.key


class Writer:
    """
    A class to write the output data to a csv file
    ...

    Attributes
    ----------
    -outpath: (str) full path of the folder in which to save the output csv file
    -full_outfilename: (str) path and name of the output csv file, including extension
    -tag_list: (list) list of all tag IDs
    -output_data: (DataFrame) pandas DataFrame of the output csv file. Columns are all tag IDs. Each column has one
        value - the correction angle of that tag. Initially, all correction angles are -1, processed tags will have
        correction angles different from -1.
    """
    def __init__(self, outpath, tag_list, outfilename="angle_correction.csv"):
        self.outpath = outpath
        self.full_outfilename = outpath + sep + outfilename
        self.tag_list = tag_list
        self.output_data = self.get_angle_correction_file()

    def get_angle_correction_file(self):
        """
        Reads data from existing angle correction file into a pandas DataFrame. If file does not exists, it first
        creates the file.
        """
        file_exists = os.path.isfile(self.full_outfilename)
        if not file_exists:
            self.create_output_file()
        output_data = pd.read_csv(self.full_outfilename)
        return output_data

    def create_output_file(self):
        """
        Creates angle correction file. Columns are all tag IDs. Each column has one value - the correction angle of that
        tag. Initializes all correction angles to be -1.:
        """
        with open(self.full_outfilename, 'w', newline='') as f:
            f_writer = csv.writer(f, delimiter=',')
            f_writer.writerow(self.tag_list)
            f_writer.writerow((-1*np.ones((1, len(self.tag_list)))).tolist()[0])

    def write_to_csv(self):
        try:
            self.output_data.to_csv(self.full_outfilename,index=False)
        except PermissionError:  # an exception that could rise if the file is open. Allows the user to close the file
            # and try to write again.
            user_input = input('PermissionError while writing to csv. \n \
            If output file is open, please close it and input [1] to try again. \n \
            Otherwise input anything else and data will not be saved')
            if user_input == '1':
                self.write_to_csv()

    def unprocessed_tags(self):
        """
        Lists all the tags whose correction angle is currently -1. These are tags that have yet to be processed and are
        passed to the main function for processing. (See 'main' function)
        """
        unprocessed_tags = self.output_data.columns[self.output_data.iloc[0] == -1]
        return unprocessed_tags

# endregion: CLASSES


def main():
    """
    The program will display random images of all tags who still don't have angle corrections (unprocessed tags), and
    ask you to click on the edge of the ants head. You may press 'Esc' to get a new random image if the displayed image
    is not good for getting the ants' heading direction. When you've clicked on the ant's head 3 times, the program will
    display several images of that ant with an arrow that should point to her head. You may approve to move on to the
    next ant, disapprove to repeat this ant, or press 'x' to approve, save the new processed tags, and exit the program.
    """

    experiment_path, output_path = path_selection_window()

    reader = TagFileReader(experiment_path,tagfile_format='python')
    writer = Writer(output_path, reader.tag_list)

    for tag_id in writer.unprocessed_tags():
        tag = Tag(tag_id, reader.tags_df, bugtag_format='python')
        if tag.random_frames is not None:  # False means that the tag wasn't recognized with high certainty for enough
            # frames, and is considered as a false recognition. In this case, skips to the next tag.
            tag.tag_loop(reader)  # main interactive loop for getting the tag's correction angle.
            writer.output_data[tag.id] = tag.correction_angle  # updating the output data after tag is processed
            if tag.user_approved == 'x':  # user chose to exit the program
                break
    writer.write_to_csv()


if __name__ == '__main__':
    main()
