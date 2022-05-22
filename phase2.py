# In[1]:


import cv2
import numpy as np 
import argparse
import time
import moviepy
from moviepy.editor import VideoFileClip
import sys

# In[4]:


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.8:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

# In[7]:


def Pipeline(frame):
    height, width, channels = frame.shape
    blob, outputs = detect_objects(frame, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    return draw_labels(boxes, confs, class_ids, classes, frame)


# In[8]:

video_input1 = VideoFileClip(sys.argv[1])
video_output1 = sys.argv[2]
processed_video = video_input1.fl_image(Pipeline)
get_ipython().run_line_magic('time', 'processed_video.set_fps(10).write_videofile(video_output1, audio=False)')
video_input1.reader.close()
video_input1.audio.reader.close_proc()


# In[ ]:

