
""" When encountering a missing frame from lower camera:
1. draw no blobs on the video
2. put text indicating missing fluorescence data on video
3. put -1 in crop data output
4. do not read frame from lower camera
5. log the event
"""


