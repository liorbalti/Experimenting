from raw_data_processing import Frame
import matplotlib.pyplot as plt
from raw_data_processing.Experiment import Experiment

path = r'Z:\Lior&Einav\Experiments\Experiment 14.0180 no larvae 21_4_19'
ex1 = Experiment(path)
trim_coordinates, frame_sizes, transformation_matrix = ex1.get_transformation_parameters()

ex1.go_to_frame(10)

_, tag_image = ex1.TagsReader.read_frame()
tag_frame = Frame.TagsFrame(tag_image)
tag_frame.full_transform(trim_coordinates,frame_sizes,transformation_matrix)

I = tag_frame.frame

plt.imshow(I)

a=1