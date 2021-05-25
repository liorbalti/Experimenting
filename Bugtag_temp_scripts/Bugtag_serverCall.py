import xmlrpc
import cv2
import os

vid_path = r'Y:\Lior&Einav\Calibration\Rhod_Atto\Ant Calibration 29_6_20\feed1\feed1vid1.avi'
vid = cv2.VideoCapture(vid_path)
_, frame = vid.read()

rs_host = '127.0.0.1'
rs_port = 8200
url = "http://%s:%d"%(rs_host,rs_port)
server = xmlrpc.client.ServerProxy(url, allow_none=True)
ver = server.GetVersion()
server.LoadProfile(r"D:\Lior\BugTag_profile_Lior.prf")
server.LoadTagFile(r"C:\BugTag\conf\tagDB_200best.csv")

tags = server.AnalFile(r'D:\Lior\forager_moreopaque.png')