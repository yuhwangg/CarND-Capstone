#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np
import time
import os

STATE_COUNT_THRESHOLD = 3

base_folder = os.path.dirname(os.path.realpath(__file__))
model_path = base_folder + '/models/frozen_sim/frozen_inference_graph.pb'
# model_path = base_folder + '/models/frozen_real/frozen_inference_graph.pb'
PATH_TO_LABELS = base_folder + '/light_classification/label_map.pbtxt'

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector', log_level=rospy.DEBUG)

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        self.current_closest_wp_index = None
        self.wp_direction = None
        self.min_wp_index = 0
        self.max_wp_index = 10901

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        # self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN

        self.previous_state = TrafficLight.UNKNOWN   # xg: code review this.

        self.last_wp = -1
        self.state_count = 0
        
        self.light_classifier = TLClassifier(model_path, PATH_TO_LABELS)

        img_full_np = self.light_classifier.load_image_into_numpy_array(np.zeros((800,600,3)))
        self.light_classifier.get_classification(img_full_np) 

        self.base_timer = time.time()

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

        light_wp, state = self.process_traffic_lights()

        # print "light_wp: " , light_wp, " state: " , state

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1



    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        min_dist = np.inf
        min_index = -1
        if self.waypoints is not None:
            for index, wp in enumerate(self.waypoints.waypoints):
                x2 = np.power(wp.pose.pose.position.x -pose.position.x , 2)
                y2 = np.power(wp.pose.pose.position.y -pose.position.y , 2)
                dist = np.sqrt(x2 + y2)
                if dist < min_dist:
                    min_dist = dist
                    min_index = index

        return min_index

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        # testing the detection result:
        # based on the test, gpu performance for one iamges is about 1 seconds, 
        # add some buffer here to avoid the images flood to the classfication
        if time.time() > self.base_timer + 0.2:
            state, d_time = self.light_classifier.get_classification(cv_image)
            self.previous_state = state
            rospy.logdebug("Detection Result: %s, time: %s" % (state, d_time))
            self.base_timer = time.time()   
            return state
        else:
            return self.previous_state


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        state = 4 # unknown

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        
        if(self.pose):
            car_wp_index = self.get_closest_waypoint(self.pose.pose)
	     
            if self.current_closest_wp_index is not None:
                if car_wp_index == self.min_wp_index and self.current_closest_wp_index ==self.maz_wp_index:
                    self.wp_direction = "increasing"
                elif car_wp_index == self.max_wp_index and self.current_closest_wp_index ==self.min_wp_index:
                    self.wp_direction = "decreasing"
                elif car_wp_index > self.current_closest_wp_index:
                    self.wp_direction = "increasing"
                elif car_wp_index < self.current_closest_wp_index:
                    self.wp_direction = "decreasing"

            self.current_closest_wp_index = car_wp_index

        #TODO find the closest visible traffic light (if one exists)

        min_dist = np.inf
        min_index = -1
        line_wp_index = -1

        for index, slp in enumerate(stop_line_positions):
            slp_pose = Pose()
            slp_pose.position.x = slp[0]
            slp_pose.position.y = slp[1]
            x2 = np.power(slp[0] - self.pose.pose.position.x ,2)
            y2 = np.power(slp[1] - self.pose.pose.position.y ,2)
            dist = np.sqrt(x2 + y2)
            if dist < min_dist:
                line_wp_index = self.get_closest_waypoint(slp_pose)

            if ( (line_wp_index >= car_wp_index and self.wp_direction=="increasing") or (line_wp_index <= car_wp_index and self.wp_direction=="decreasing") ) and  dist<120:
                min_dist = dist
                min_index = index
                light = self.lights[min_index]

        if light:
            # This line should be uncommented once the "get_light_state" is ready 
            state = self.get_light_state(light)
            # state = light.state
            
            return line_wp_index, state
          
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
        # # activate the tensorflow  --- this does not work
        # rospy.logdebug("test images, loading")
        # tt_image = cv2.imread(test_image)
        # print(tt_image)
        # self.light_classifier.get_classification(tt_image)
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
