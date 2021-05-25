from raw_data_processing import Experiment
from os import sep as sep

expName = '4_19_5_19'
path = r'Z:\Lior&Einav\Experiments\Experiment_'+expName+'\With food'
outpath = path + sep + 'blob analysis normalized'
ex1 = Experiment.Experiment2Colors(path, get_timestamps=True)
trim_coordinates, frame_sizes, transformation_matrix = ex1.get_transformation_parameters()