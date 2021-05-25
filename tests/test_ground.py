# import Experiment
# import numpy as np
# import csv
# import Reader
#
# from os import sep as sep
#
#
# expName = '8_18_6_19'
# path = r'Z:\Lior&Einav\Experiments\Experiment_'+expName+'\With food\DXgrabber'
# #ex1 = Experiment.Experiment2Colors(path)
#
# tagsReader = Reader.VideoReader(path,'txt',get_timestamps=True)
# timestamps = tagsReader.get_timestamps()
#
# a=1

import tkinter as tk
import tkinter.filedialog


def get_experiment_folder():
    path = tk.filedialog.askdirectory()
    experiment_path.set(path)


def get_output_folder():
    path = tk.filedialog.askdirectory()
    output_path.set(path)


window = tk.Tk()
window.title("Angle correction")

experiment_path = tk.StringVar()
output_path = tk.StringVar()

tk.Label(window,text='Please select the following paths').grid(row=0,column=2,padx=20,pady=20)
tk.Label(window,text='Experiment path:').grid(row=1,column=1,padx=20)
tk.Label(window,text='Output path:').grid(row=2,column=1,padx=20,pady=20)

e1 = tk.Entry(window, textvariable=experiment_path, width=80).grid(row=1, column=2)
e2 = tk.Entry(window, textvariable=output_path, width=80).grid(row=2, column=2, pady=20)

b1 = tk.Button(window,text='Select experiment folder', command=get_experiment_folder)
b2 = tk.Button(window,text='Select output folder', command=get_output_folder)
b1.grid(row=1,column=3,padx=20)
b2.grid(row=2,column=3,padx=20,pady=20)

b3 = tk.Button(window,text='OK',command=window.destroy).grid(row=3,column=2,pady=20)

window.mainloop()


a = 1
