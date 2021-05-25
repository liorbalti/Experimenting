import os, shutil
from raw_data_processing import Reader
import re
sep = os.sep

augmented_library_path = r'D:\Nadav\trophallaxisTRAIN'
output_path = r'D:\Lior\NadavLib_noAugmentation'
reader = Reader.FileReader(augmented_library_path, 'tif')

# create subfolders in output_path
for subfolder in reader.subfolder_list:
    os.mkdir(output_path + sep + subfolder)

num_files = len(reader.file_list)
cond_reg = r"(_[1-3](_ref[xy])?\.tif)$"
for c, f in enumerate(reader.file_list):
    augmented = re.search(cond_reg,f)
    if augmented is None:
        split_path = f.split(sep)
        foldername = split_path[-2]
        shutil.copy(f, output_path + sep + foldername)
        print(f"copying file {c} out of {num_files}")
print("Done!")
