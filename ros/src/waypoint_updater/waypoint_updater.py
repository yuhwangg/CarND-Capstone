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
import matplotlib.pyplot as plt

# FIXME ::<xg>:: this is only for testing purpose, delete this.
from styx_msgs.msg import TrafficLightArray, TrafficLight  

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

# <xg>
# per definition in the /waypoint_loader/launch/*.launch files 
# for the simulator, the speed limit is 40 kmph, the site limit is 10 kmph
# this value should be changed. 
MAX_SPEED = 40  
MAX_DECEL = 1.0 
# </xg>


# set the value to True to enable the velocity update:
UPDATE_VELOCITY = True
USING_SIMULATE_DATA = False

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=rospy.DEBUG)

        self.pose = None
        self.waypoints = None
        self.current_velocity = None
        self.last_waypoint_index = None
        self.waypoint_direction = None

        # Subsciber
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        #rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.wp_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('current_velocity', TwistStamped, self.velocity_cb)
        # rospy.Subscriber('/obstacle_waypoint', Int32, obstacle_cb)

        # FIXME ::<xg>:: this is only for testing purpose, delete this.
        if USING_SIMULATE_DATA:
            rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
            self.traffic_lights_wp_mapping = None
        else:
            rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # the next wayponint index, this is an indication of the car's position
        # kind of like the "localization" result. 
        self.next_waypoint_index = None 

        # the waypoint index to the closest red light. published by the tl_detector
        self.traffic_waypoint = None
        # </xg>

        # Publisher
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.final_waypoints_index_pub = rospy.Publisher('final_index', Int32, queue_size=1)
        self.cte_pub = rospy.Publisher('/vehicle/cte', Float32, queue_size=1)
        self.round = 0
        self.debug_cte = False

        if self.debug_cte:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)  # 0.5Hz
        while not rospy.is_shutdown():
            if (self.pose is not None) and (self.waypoints is not None):
                self.update_final_waypoints()
                self.publish_cte()
                self.update_velocity(self.final_waypoints) # <xg>: update the velocity
                self.publish_final_waypoints()

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
        if USING_SIMULATE_DATA:
            # here the msg is from /vehicle/traffic_lights
            self.traffic_waypoint = self._get_red_light_wp_index(msg)
        else:
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
        map_coords = self.waypoints[index].pose.pose.position

        map_x = map_coords.x
        map_y = map_coords.y

        heading = math.atan2(map_y - position.y, map_x - position.x)
        angle = math.fabs(theta - heading)
        if angle > math.pi / 4:
            index += 1
        index = self.norm_index(index)

        return index

    def norm_index(self,index):
        wp_count = len(self.waypoints)
        index = abs(index % wp_count)
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
        self.next_waypoint_index = index # <xg: added for simu TL status>
        final_waypoints = []
        len1 = len(self.waypoints)
        for i in range(LOOKAHEAD_WPS):
            wp = (i + index) % len1
            waypoint = copy.deepcopy(self.waypoints[wp])
            final_waypoints.append(waypoint)
        self.final_waypoints = final_waypoints

    def update_velocity(self, waypoints):
        if not UPDATE_VELOCITY:
            return

        if self.traffic_waypoint is None:
            return 
        
        if self.traffic_waypoint < 0: # there is no red light, we speed up
            max_speed_mps = MAX_SPEED * 1000 / 3600   # convert km per hour to m per second
            for i in range(len(waypoints)):
                self.set_waypoint_velocity(waypoints, i, max_speed_mps)
        else:    # decrease the velocity
            # get the index of traffic light waypoint in the waypoints
            rospy.logdebug("Vupd-traffic_waypoint: (%s, %s, %s)" % (self.traffic_waypoint, self.waypoints[self.traffic_waypoint].pose.pose.position.x, self.waypoints[self.traffic_waypoint].pose.pose.position.y ))
            rospy.logdebug("Vupd-next_waypoint_index: (%s, %s, %s)" % (self.next_waypoint_index, self.waypoints[self.next_waypoint_index].pose.pose.position.x, self.waypoints[self.next_waypoint_index].pose.pose.position.y ))
            
            relative_tl_wp_index = self.traffic_waypoint - self.next_waypoint_index
            rospy.logdebug("Vupd-Relative red light index %s" % relative_tl_wp_index)
            # test, the index conversion is corret
            try:
                assert(self.distance_2d(waypoints[relative_tl_wp_index].pose.pose.position, self.waypoints[self.traffic_waypoint].pose.pose.position) < 0.0001)
            except AssertionError:
                rospy.logerr("Error-Vupd-AssertionError. ")
                # rospy.logdebug("Vupd-relative_tl_wp_index: (%s, %s, %s)" % (relative_tl_wp_index, waypoints[relative_tl_wp_index].pose.pose.position.x, waypoints[relative_tl_wp_index].pose.pose.position.y ))
                return 
            except IndexError:
                rospy.logerr("Error-Vupd-IndexError. ")
                return

            total_distance_to_stop = self.wp_distance(waypoints, 0, relative_tl_wp_index)
            rospy.logdebug("Vupd-The total distance to stop: %.4f" % total_distance_to_stop)
            self.set_waypoint_velocity(waypoints, relative_tl_wp_index, 0)   # set the velocity of last waypoint is zero
            
            # linearly decrease the other velocity, based on the distanct to stop line
            # initial_velocity = self.get_waypoint_velocity(waypoints[0])
            initial_velocity = self.current_velocity.linear.x
            rospy.logdebug("Vupd-The initial velocity %s" % initial_velocity)
            dist = 0
            for i in range(relative_tl_wp_index-1):
                dist += self.distance_2d(waypoints[i].pose.pose.position, waypoints[i+1].pose.pose.position)
                v = initial_velocity * (1 - float(dist) / total_distance_to_stop)
                if v < 1.0:
                    v = 0.
                self.set_waypoint_velocity(waypoints, i, v)

            # last = waypoints[relative_tl_wp_index]
            # last.twist.twist.linear.x = 0.
            # for wp in waypoints[:relative_tl_wp_index][:-1][::-1]:
            #     dist = self.distance_2d(wp.pose.pose.position, last.pose.pose.position)
            #     rospy.logdebug("Vupd-dist %s" % dist)
            #     vel = math.sqrt(2 * MAX_DECEL * dist)
            #     rospy.logdebug("Vupd-dist %s" % vel)
            #     if vel < 1.:
            #         vel = 0.
            #     wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            # the above code is from the "decelerate()" method from the "waypoint_loader" file.
        rospy.logdebug("Vupd- The updated velocity: %s" % (",".join(["%.2f" % wp.twist.twist.linear.x for wp in waypoints])))
        #rospy.logdebug("Next waypoint velocity: %s" % ( waypoints[0].twist.twist.linear.x ))
        pass


    def publish_final_waypoints(self):
        msg = Lane()
        msg.header.stamp = rospy.Time(0)
        msg.waypoints = self.final_waypoints
        self.final_waypoints_pub.publish(msg)

    def world_to_car_coords(self, origin, point, angle):

        angle = -angle + math.pi / 2 #adjust simulator angle


        while angle < -math.pi or angle > math.pi: #normalizing
            #print("Normalizing: ",angle,"\n")
            angle += 2 * math.pi * (1 if angle < -math.pi else -1)


        px, py = point.x-origin.x, point.y-origin.y
        angle_sin = np.sin(angle)
        angle_cos = np.cos(angle)
        x = angle_cos * px - angle_sin * py
        y = angle_sin * px + angle_cos * py
        #print(angle, x,y)
        return x, y

    def publish_cte(self):
        msg = Float32()
        car_position = self.pose.pose.position
        car_yaw = self.get_current_yaw()
        index = self.next_waypoint(car_position, car_yaw)
        cnt = 6

        x_list = []
        y_list = []
        indexes = []
        for i in range(cnt):
            wp_index = self.norm_index(index + i - cnt/2)
            indexes.append(wp_index)
            point = self.waypoints[wp_index].pose.pose.position
            x,y = self.world_to_car_coords(car_position, point, car_yaw)
            x_list.append(x)
            y_list.append(y)



        coeffs = np.polyfit(y_list,x_list,2)
        cte = coeffs[-1] # fit for x = 0
        #print('indexes:', indexes, "CTE:", cte)
        #print('car_position:', car_position.x, car_position.y)

        msg.data = cte
        self.cte_pub.publish(msg)

        if self.debug_cte:
            self.show_cte(x_list,y_list)


    def show_cte(self,x_list,y_list):
        #introduces a systematic delay for rendering, it can affect the PID controller
        self.ax.clear()
        self.ax.plot(x_list, y_list, 'ro')
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['bottom'].set_position('zero')
        plt.axis('equal')
        plt.ylim([-30, 30])
        plt.xlim([-10, 10])

        plt.draw()
        plt.pause(0.0000000001)


    # FIXME ::<xg>:: this is only for testing purpose, delete this.
    # this method uses the simulator's TrafficLightArray message to detect 
    # the index of the clostest way-point to the red light, if no red light detected, -1 returns: 
    # https://carnd.slack.com/archives/G83RE7171/p1511918235000351
    def _get_red_light_wp_index(self, tl_array_msg):

        if self.next_waypoint_index is None:
            return

        rospy.logdebug("TL-Info: %s " % (",".join(["(%s, %s, %s, %s)" % (i, tl.pose.pose.position.x, tl.pose.pose.position.y, tl.state) for i, tl in enumerate(tl_array_msg.lights)])))

        def map_traffic_lights_to_waypoints(traffic_lights):
            """ map the traffic lights location to the waypoint.
                this is done only once. 
            """
            self.traffic_lights_wp_mapping = []
            for tl_index, tl in enumerate(traffic_lights.lights):
                nearest_index = 0
                nearest_dist = self.distance_2d(tl.pose.pose.position, self.waypoints[0].pose.pose.position)
                for i in range(1, len(self.waypoints)):
                    dist = self.distance_2d(tl.pose.pose.position, self.waypoints[i].pose.pose.position)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_index = i
                self.traffic_lights_wp_mapping.append(nearest_index)

            rospy.logdebug("TL-WP mapping: %s" % ",".join([str(l) for l in self.traffic_lights_wp_mapping])) 

        # mapping the traffic light to waypoint, only done once
        if self.traffic_lights_wp_mapping is None:
            map_traffic_lights_to_waypoints(tl_array_msg)

        # find the traffic lights, based on two creteria:
        # 1. the state is red or yellow
        # 2. its' indices must fall in range [next_wp: next_wp + lookahead points]
        rospy.logdebug("TL-Lookahead index (%s, %s)" % (self.next_waypoint_index, self.next_waypoint_index+LOOKAHEAD_WPS))
        red_lights_index = []
        for tl_index, tl in enumerate(tl_array_msg.lights):
            # fixbug: exclude the points in the ends.
            # if the car is already in the red light location, keep moving;
            # if the car is in the last lookahead way points, we ignore it. since it still far away. 
            # if self.traffic_lights_wp_mapping[tl_index] >= self.next_waypoint_index and self.traffic_lights_wp_mapping[tl_index] <= self.next_waypoint_index + LOOKAHEAD_WPS and (tl.state == TrafficLight.RED or tl.state == TrafficLight.YELLOW):
            if self.traffic_lights_wp_mapping[tl_index] > self.next_waypoint_index and self.traffic_lights_wp_mapping[tl_index] < self.next_waypoint_index + LOOKAHEAD_WPS and (tl.state == TrafficLight.RED or tl.state == TrafficLight.YELLOW):
                red_lights_index.append(self.traffic_lights_wp_mapping[tl_index])
                rospy.logdebug("TL-The red light array index %s " % tl_index)
        rospy.logdebug("TL-Red Light index: %s" % ",".join(["(%s, %s, %s)" % (l, self.waypoints[l].pose.pose.position.x, self.waypoints[l].pose.pose.position.y) for l in red_lights_index]))
        # if there are multiple red lights in range, returns the closest one only.
        red_light_wp_index = -1
        if len(red_lights_index) > 0:
            red_light_wp_index = min(red_lights_index)
        rospy.logdebug("TL-Closest RL index %s" % red_light_wp_index)
        return red_light_wp_index


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
