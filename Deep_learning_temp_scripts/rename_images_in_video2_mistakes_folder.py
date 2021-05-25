import os
from raw_data_processing import Reader
from os import sep as sep

path = r'F:\Lior\Network mistakes experiment 9\network_mistakes_exp9v2'
reader = Reader.FileReader(path, 'tif')

for filename in reader.file_list:
    subfolder = str.split(filename, sep)[-2]
    old_name = str.split(filename, sep)[-1]
    new_name = old_name[:5] + '2' + old_name[6:]

    dst = path + sep + subfolder + sep + new_name

    os.rename(filename,dst)

a=1
