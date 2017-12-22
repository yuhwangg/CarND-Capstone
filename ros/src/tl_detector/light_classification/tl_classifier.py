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

        self.detection_graph = None
        self.tf_session = None
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.num_detections = None

        # self.load_graph()

    # Helper function to convert image into numpy array
    def load_image_into_numpy_array(self, image):
         return np.asarray(image, dtype="uint8" )

    def load_graph(self):
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

        if self.tf_session is None:
            self.load_graph()
            self.tf_session = tf.Session(graph=self.detection_graph)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        image_np_expanded = np.expand_dims(image, axis=0)
        detect_time = time.time()
        (boxes, scores, classes, num) = self.tf_session.run(
              [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
              feed_dict={self.image_tensor: image_np_expanded})

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
