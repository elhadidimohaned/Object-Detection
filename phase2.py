# In[1]:


import cv2
import numpy as np 
import argparse
import time
import moviepy
from moviepy.editor import VideoFileClip
import sys

# In[2]:


def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()] 
        
    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    return net, classes, output_layers


# In[3]:


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs



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


# In[5]:


def draw_labels(boxes, confs, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.65, 0.7)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)
            cv2.putText(img, label, (x, y - 5), font, 0.5, (0,139,139), 2)
    return img


# In[6]:


model, classes, output_layers = load_yolo()




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


