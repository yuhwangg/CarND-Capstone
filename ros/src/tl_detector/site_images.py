import rospy
import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
from glob import glob
import csv

def create_dataset():
    resize_scale = 0.25 # /4 = 342x274
    img_format = 'jpg'
    src = 'traffic_light_bag_files/img/'
    dst = 'traffic_light_bag_files/dataset/'
    dataset_name = 'dataset.csv'
    lbls = {0: 'red', 1: 'yellow', 2: 'green', 3: 'unknown'} # match TrafficLight.msg

    dataset = []
    for label in lbls:
        name = lbls[label]
        src_path = "{}{}/*.png".format(src,name)
        for filepath in glob(src_path):
            filename_ext = filepath.split("/")[3]
            filename = filename_ext.split('.')[0]
            dst_path_rel = "img/{}.{}".format(filename,img_format) # used in dataset.csv
            dst_path = "{}/{}".format(dst,dst_path_rel)
            img = cv2.imread(filepath)
            height, width = img.shape[:2]
            img_res = cv2.resize(img, ( (int)(resize_scale*width), (int)(resize_scale*height)) )
            cv2.imwrite(dst_path,img_res)
            dataset.append( (dst_path_rel, label))

    with open(dst+dataset_name,'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_ALL)
        for row in dataset:
            csvwriter.writerow(row)




def parse_bag():
    src = "traffic_light_bag_files/just_traffic_light.bag"
    src = "traffic_light_bag_files/loop_with_traffic_light.bag"
    dst = "traffic_light_bag_files/img/"
    bridge = CvBridge()
    for topic, msg, t in rosbag.Bag(src).read_messages():
        if topic == '/image_raw':
            name = "{}{}.png".format(dst,t)
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite(name, cv_image)
            print(len(msg.data))

# parse_bag() #call to extract images from bags
create_dataset() #call to generate dataset
