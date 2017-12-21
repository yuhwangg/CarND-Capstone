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

sim_model_path = '/home/zhangxg/work/vm_share/ros/catkin_ws/src/CarND-Capstone/ros/src/tl_detector/models/frozen_sim/frozen_inference_graph.pb'

PATH_TO_LABELS = '/home/zhangxg/work/vm_share/ros/catkin_ws/src/CarND-Capstone/ros/src/tl_detector/light_classification/label_map.pbtxt'
NUM_CLASSES = 14
min_score_thresh = 0.5


from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.load_graph()

    # def load_image_into_numpy_array(self, image):
    #   (im_width, im_height) = image.size
    #   return np.array(image.getdata()).reshape(
    #       (im_height, im_width, 3)).astype(np.uint8)

    def load_graph(self,):
      self.detection_graph = tf.Graph()
      with self.detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(sim_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction  
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

                (boxes, scores, classes, num) = sess.run(
                      [detection_boxes, detection_scores, detection_classes, num_detections],
                      feed_dict={image_tensor: image_np_expanded})

                for i in range(boxes.shape[0]):
                    if scores is None or np.any(scores[i] > min_score_thresh):
                        class_name = self.category_index[classes[i]]['name']
                        rospy.logdebug('{}'.format(class_name), scores[i])
                        # if class_name == "Red":
                        #     return TrafficLight.RED
                        # elif class_name == "Yellow":
                        #     return TrafficLight.YELLOW
                        # elif class_name == "Green":
                        #     return TrafficLight.GREEN
                        # else:
                        #     return TrafficLight.UNKNOWN
