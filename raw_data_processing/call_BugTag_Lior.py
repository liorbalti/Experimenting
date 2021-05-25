# check parameters
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
from copy import copy
from os import sep as sep
import xmlrpc.client
import subprocess
from raw_data_processing import Reader

# Definitions
# images_path = r'F:\Lior\Deep learning\Matlab\Deep Learning scripts\Classifying\for paper\images for example'
# output_path = images_path
# add_numbers=True
images_path = r'Y:\Lior&Einav\Experiments\experiment17_100920\with food\DXgrabber'   # path to folder of images
output_path = r'Y:\Lior&Einav\Experiments\experiment17_100920\with food\Bugtag output'
add_numbers = True  # option to write tag numbers on the image
show_preview = False  # option to preview the tagged image
create_video = True  # option to create a video

# BugTag Parameters
bugtag_host = "127.0.0.1"
bugtag_port = 8200
bugtag_DB = r"C:\BugTag\conf\tagDB_300best.csv"
bugtag_profile = r"D:\Lior\phd\BugTag_profile_Lior.prf"


class BugTag_server():
    def __init__(self, host, port):
        self.server = None
        self.host = host
        self.port = port

    def connect(self, max_attempts=10, exe_file=r'C:\BugTag\bin\BugTag.exe'):
        subprocess.Popen([exe_file])
        url = "http://%s:%d" % (self.host, self.port)
        connected = False
        attempt_counter = 1
        while connected is False:
            self.server = xmlrpc.client.ServerProxy(url, allow_none=True)
            try:
                ver = self.server.GetVersion()
                print("Connected to BugTag. Version " + str(ver))
                return
            except:
                if attempt_counter > max_attempts:
                    print("Failed to connect to BugTag.")
                    break
            attempt_counter += 1

    def load_tagDB(self, tagDB_file):
        self.server.LoadTagFile(tagDB_file)

    def load_profile(self, profile_file):
        self.server.LoadProfile(profile_file)

    def setup(self, tagDB_file=None, profile_file=None):
        self.connect()
        if tagDB_file is not None:
            self.load_tagDB(tagDB_file)
            print("loaded tag DB from "+ tagDB_file)
        if profile_file is not None:
            self.load_profile(profile_file)
            print("loaded profile from " + profile_file)

    def tag_image(self, image_file):
        tags_info = self.server.AnalFile(image_file)
        return tags_info


def writedata(file_name, array):
    with open(file_name, 'a') as g:
        writer3 = csv.writer(g, delimiter=',', lineterminator='\n')
        writer3.writerows(array)


class OutputData():
    def __init__(self, output_path, output_filename):
        self.output_path = output_path
        self.output_filename = output_filename
        self.full_output_path = output_path + sep + output_filename
        self.line = 0
        self.tagline = None
        self.antNum = None   # dictionary pointing from each tag number to its first column in the output data
        self.names = None    # all tags from DB

    def create_antNum(self, antnames_file=r"Y:\\honeypot\\honeypot\\antgranary\\Lior\\Bugtag\\bugtag_new2\\tagDB_300best.txt"):
        names = np.genfromtxt(antnames_file, delimiter=',', dtype=np.int32)
        antNum = {}
        j = 2
        for n in names:
            antNum[n] = j
            j += 4
        self.antNum = antNum
        self.names = names

    def write_header(self):
        if self.antNum is None:
            self.create_antNum()
        output_header = ['frame #', ' recognitions']
        for n in self.names:
            output_header += [str(n) + '-x', str(n) + '-y', str(n) + '-angle', str(n) + '-error']
        writedata(self.full_output_path, [output_header])

    def arrange_data(self,taginfo):
        output_data = -1 * np.ones((1, 4 * 300 + 2))
        output_data[0][0] = self.line
        output_data[0][1] = len(taginfo)
        for tag in taginfo:
            output_data[0][self.antNum[tag[0]]] = tag[1][0]  # x
            output_data[0][self.antNum[tag[0]] + 1] = tag[1][1]  # y
            output_data[0][self.antNum[tag[0]] + 2] = tag[2]  # angle
            output_data[0][self.antNum[tag[0]] + 3] = tag[3]  # tag error
        self.tagline = output_data

    def write_tagline(self,taginfo):
        self.arrange_data(taginfo)
        writedata(self.full_output_path, self.tagline)


class OutputVideo():
    def __init__(self, output_path, output_vidname, add_numbers=False, show_preview=False):
        self.output_path = output_path
        self.output_filename = output_vidname
        self.full_output_path = output_path + sep + output_vidname
        self.image = None
        self.image_with_tags = None
        self.height = None
        self.width = None
        self.outMovie = None
        self.outMovie_with_tags = None
        self.add_numbers = add_numbers   # whether or not to insert tag numbers on image
        self.show_preview = show_preview   # whether or not to preview the tagged image
        self.line = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def read_image(self,image_file):
        self.image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if self.height is None:
            self.height, self.width = self.image.shape

    def add_tags_to_image(self,taginfo):
        self.image_with_tags = cv2.cvtColor(copy(self.image), cv2.COLOR_GRAY2RGB)
        for tag in taginfo:
            cv2.putText(self.image_with_tags, str(tag[0]), (int(tag[1][0]), int(tag[1][1])), self.font, 1.9, (0, 240, 255), 3)

    def preview(self):
        if self.line == 0:
            plt.figure(1)
            self.prev_img = plt.imshow(self.image_with_tags)
        else:
            plt.figure(1)
            self.prev_img.set_data(self.image_with_tags)
            plt.pause(0.01)
            plt.draw()

    def write_frame(self):
        if len(self.image.shape) == 3:
            self.outMovie.write(cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        elif len(self.image.shape) == 2:
            self.outMovie.write(cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR))

    def write_frame_with_tags(self):
        self.outMovie_with_tags.write(cv2.cvtColor(self.image_with_tags, cv2.COLOR_RGB2BGR))

    def create_outVideo(self):
        self.outMovie = cv2.VideoWriter(self.full_output_path, self.fourcc, 20.0,
                                        (self.width, self.height), True)

    def create_outVideo_with_tags(self):
        self.outMovie_with_tags = cv2.VideoWriter(self.output_path + sep + 'with_tags_' + self.output_filename, self.fourcc, 20.0,
                                        (self.width, self.height), True)

    def do_line(self,taginfo):
        if self.outMovie is None:
            self.create_outVideo()
        self.write_frame()
        if self.add_numbers:
            if self.outMovie_with_tags is None:
                self.create_outVideo_with_tags()
            self.add_tags_to_image(taginfo)
            self.write_frame_with_tags()
            if self.show_preview:
                self.preview()


def main():
    bugtag = BugTag_server(bugtag_host,bugtag_port)
    bugtag.setup(tagDB_file=bugtag_DB, profile_file=bugtag_profile)

    reader = Reader.FileReader(images_path, extension='tif')
    prev_subfoldername = ''

    # loop over all image files
    for file in reader.file_list:

        split_path = file.split(sep)
        subfoldername = split_path[-2]

        #  if changed subfolder, create new output file and output video
        if subfoldername != prev_subfoldername:
            output_filename = subfoldername + ".csv"
            output_vidname = "Tagged_line-1" + subfoldername + ".avi"

            output_data = OutputData(output_path, output_filename)
            output_data.write_header()

            if create_video:
                output_video = OutputVideo(output_path, output_vidname, add_numbers=add_numbers, show_preview=show_preview)

        print(subfoldername + " " + str(output_data.line))

        taginfo = bugtag.tag_image(file)
        output_data.write_tagline(taginfo)
        output_data.line += 1

        if create_video:
            output_video.read_image(file)
            output_video.do_line(taginfo)
            output_video.line += 1

        prev_subfoldername = subfoldername

        # if output_data.line > 5:
        #     break

    output_video.outMovie.release()


main()
