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

