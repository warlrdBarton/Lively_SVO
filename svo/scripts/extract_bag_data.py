import sys
print(sys.executable)

import os
import rosbag
import cv2
from cv_bridge import CvBridge
import csv
import sensor_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, Imu
import tf

bag_path = "/home/sunteng/dataset/day_11_25_office_stereo_imu_sync_0_640_480.bag"
image_folder = "/home/sunteng/dataset/day_11_25_office_stereo_imu_sync_0_640_480/data"
csv_file = "/home/sunteng/dataset/day_11_25_office_stereo_imu_sync_0_640_480/imu_data.csv"
img_file = "/home/sunteng/dataset/day_11_25_office_stereo_imu_sync_0_640_480/img_info.txt"


os.makedirs(image_folder, exist_ok=True)


img_info_list=[]

bridge = CvBridge()
idx=0
with rosbag.Bag(bag_path, 'r') as bag, open(csv_file, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "orientation.x", "orientation.y", "orientation.z", "orientation.w",
                     "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
                     "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"])
    
    for topic, msg, t in bag.read_messages():
        # print(topic,type(msg),t)

        if topic=="/camera/infra2/image_rect_raw":
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            img_filename = os.path.join(image_folder, str(t.to_nsec()) + "_r.png")
            img_info_list.append(str(idx)+' '+str(t)+' '+str(t.to_nsec()) + "_r.png")
            idx+=1
            cv2.imwrite(img_filename, cv_img)
        elif topic=="/camera/infra1/image_rect_raw":
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            img_filename = os.path.join(image_folder, str(t.to_nsec()) + "_l.png")
            img_info_list.append(str(idx)+' '+str(t)+' '+str(t.to_nsec()) + "_l.png")
            idx+=1
            cv2.imwrite(img_filename, cv_img)
        elif topic=="/camera/imu":
            writer.writerow([t.to_nsec(), msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
                             msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                             msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])


print(img_info_list[0])
with open(img_file, 'w') as f:
    # 将int，float，str数据写入文件
    for i in img_info_list:
        f.write(i +'\n')