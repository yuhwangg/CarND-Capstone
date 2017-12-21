import numpy as np
import os
import sys
import tensorflow as tf
import time

from collections import defaultdict
from io import StringIO
from PIL import Image

import label_map_util
import visualization_utils as vis_util

NUM_CLASSES = 14
min_score_thresh = 0.5

from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self, model_path, label_path):
        #TODO load classifier
        self.label_map = label_map_util.load_labelmap(label_path)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.model_path = model_path
        self.load_graph()

    def load_graph(self,):
      self.detection_graph = tf.Graph()
      with self.detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(self.model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    def get_classification(self, image):
        light_status = TrafficLight.UNKNOWN
        if image is None:
            return light_status 

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                image_np_expanded = np.expand_dims(image, axis=0)
                detect_time = time.time()
                (boxes, scores, classes, num) = sess.run(
                      [detection_boxes, detection_scores, detection_classes, num_detections],
                      feed_dict={image_tensor: image_np_expanded})

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                for i in range(boxes.shape[0]):
                    if scores is None or scores[i] > min_score_thresh:
                        class_name = self.category_index[classes[i]]['name']
                        if class_name == "Red":
                            light_status =  TrafficLight.RED
                        elif class_name == "Yellow":
                            light_status =  TrafficLight.YELLOW
                        elif class_name == "Green":
                            light_status =  TrafficLight.GREEN
                        else:
                            light_status =  TrafficLight.UNKNOWN
                    return light_status, (time.time() - detect_time)