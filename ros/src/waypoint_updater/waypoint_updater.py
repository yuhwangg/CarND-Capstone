#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32, Float32
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint

import math
import tf
import copy
import numpy as np
from scipy import interpolate

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.pose = None
        self.waypoints = None
        self.current_velocity = None

        # Subsciber
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        #rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.wp_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('current_velocity', TwistStamped, self.velocity_cb)
        # rospy.Subscriber('/obstacle_waypoint', Int32, obstacle_cb)

        # Publisher
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.final_waypoints_index_pub = rospy.Publisher('final_index', Int32, queue_size=1)
        self.cte_pub = rospy.Publisher('/vehicle/cte', Float32, queue_size=1)

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)  # 0.5Hz
        while not rospy.is_shutdown():
            if (self.pose is not None) and (self.waypoints is not None):
                self.update_final_waypoints()
                self.publish_final_waypoints()
                self.publish_cte()
            rate.sleep()
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
        pass

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
        # is only need once - so better unregister it
        self.wp_sub.unregister()
        pass

    def traffic_cb(self, msg):
        self.traffic_waypoint = msg.data
        # TODO: We surely need to do more here
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        self.obstacle_waypoint = msg
        pass

    def velocity_cb(self, msg):  # geometry_msgs/TwistStamped
        self.current_velocity = msg.twist

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def wp_distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        len_waypoints = len(waypoints)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def distance_2d(self, a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def closest_waypoint(self, position):
        closest_len = 10000
        closest_index = 0
        for i in range(len(self.waypoints)):
            dist = self.distance_2d(position, self.waypoints[i].pose.pose.position)
            if dist < closest_len and dist >= 0:
                closest_len = dist
                closest_index = i

        return closest_index

    def next_waypoint(self, position, theta):
        index = self.closest_waypoint(position)
        map_x = self.waypoints[index].pose.pose.position.x
        map_y = self.waypoints[index].pose.pose.position.y

        heading = math.atan2(map_y - position.y, map_x - position.x)
        angle = math.fabs(theta - heading)
        if angle > math.pi / 4:
            index += 1

        return index

    def get_current_yaw(self):
        orientation = [
            self.pose.pose.orientation.x,
            self.pose.pose.orientation.y,
            self.pose.pose.orientation.z,
            self.pose.pose.orientation.w]
        euler = tf.transformations.euler_from_quaternion(orientation)
        return euler[2]  # z direction

    def update_final_waypoints(self):
        theta = self.get_current_yaw()
        index = self.next_waypoint(self.pose.pose.position, theta)
        final_waypoints = []
        len1 = len(self.waypoints)
        for i in range(LOOKAHEAD_WPS):
            wp = (i + index) % len1
            waypoint = copy.deepcopy(self.waypoints[wp])
            final_waypoints.append(waypoint)

        self.final_waypoints = final_waypoints
        self.final_waypoints_index_pub.publish(Int32(index))

    def publish_final_waypoints(self):
        msg = Lane()
        msg.header.stamp = rospy.Time(0)
        msg.waypoints = self.final_waypoints
        self.final_waypoints_pub.publish(msg)

    def world_to_car_coords(self, origin, point, angle):
        px, py = point.x-origin.x , point.y-origin.y
        angle_sin = math.sin(angle)
        angle_cos = math.cos(angle)
        x = angle_cos * px - angle_sin * py
        y = angle_sin * px + angle_cos * py
        return x, y

    def publish_cte(self):
        msg = Float32()
        car_position = self.pose.pose.position
        car_yaw = self.get_current_yaw()
        index = self.next_waypoint(car_position, car_yaw)

        next_point = self.waypoints[index].pose.pose.position
        prev_point = self.waypoints[index-1].pose.pose.position

        #print(car_yaw)
        #print(prev_point.x, prev_point.y)
        #print(car_position.x, car_position.y)
        #print(next_point.x, next_point.y)

        x1,y1 = self.world_to_car_coords(car_position, next_point, car_yaw)
        x2,y2 = self.world_to_car_coords(car_position, prev_point, car_yaw)

        #print(x1, y1)
        #print(x2, y2)

        coeffs = np.polyfit([x1,x2],[y1,y2],1)
        cte = coeffs[-1] # fit for x = 0
        print('CTE:',cte)

        msg.data = cte
        self.cte_pub.publish(msg)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
