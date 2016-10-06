#!/usr/bin/python
"""
view_rosbag_video.py: version 0.1.0
Note:
Part of this code was copied and modified from github.com/comma.ai/research (code: BSD License)

Todo:
Update steering angle projection.  Current version is a hack from comma.ai's version
Update enable left, center and right camera selection.  Currently all three cameras are displayed.
Update to enable display of trained steering data (green) as compared to actual (blue projection).

History:
2016/10/06: Update to add --skip option to skip the first X seconds of data from rosbag.
2016/10/02: Initial version to display left, center, right cameras and steering angle.
"""

import argparse
import sys
import numpy as np
import pygame
import rosbag
import datetime

#from keras.models import model_from_json

pygame.init()
size = (320*3, 240)
pygame.display.set_caption("Udacity SDC challenge 2: camera video viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)

imgleft = pygame.surface.Surface((320,240),0,24).convert()
imgcenter = pygame.surface.Surface((320,240),0,24).convert()
imgright = pygame.surface.Surface((320,240),0,24).convert()

# ***** get perspective transform for images *****
from skimage import transform as tf

rsrc = \
 [[43.45456230828867, 118.00743250075844],
  [104.5055617352614, 69.46865203761757],
  [114.86050156739812, 60.83953551083698],
  [129.74572757609468, 50.48459567870026],
  [132.98164627363735, 46.38576532847949],
  [301.0336906326895, 98.16046448916306],
  [238.25686790036065, 62.56535881619311],
  [227.2547443287154, 56.30924933427718],
  [209.13359962247614, 46.817221154818526],
  [203.9561297064078, 43.5813024572758]]
rdst = \
 [[10.822125594094452, 1.42189132706374],
  [21.177065426231174, 1.5297552836484982],
  [25.275895776451954, 1.42189132706374],
  [36.062291434927694, 1.6376192402332563],
  [40.376849698318004, 1.42189132706374],
  [11.900765159942026, -2.1376192402332563],
  [22.25570499207874, -2.1376192402332563],
  [26.785991168638553, -2.029755283648498],
  [37.033067044190524, -2.029755283648498],
  [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))

def perspective_tform(x, y):
  p1, p2 = tform3_img((x,y))[0]
  return p2, p1

# ***** functions to draw lines *****
def draw_pt(img, x, y, color, shift_from_mid, sz=1):
  col, row = perspective_tform(x, y)
  row = int(row) + shift_from_mid
  col = int(col+img.get_height()*2)/3
  if row >= 0 and row < img.get_width()-sz and\
     col >= 0 and col < img.get_height()-sz:
    img.set_at((row-sz,col-sz), color)
    img.set_at((row-sz,col), color)
    img.set_at((row-sz,col+sz), color)
    img.set_at((row,col-sz), color)
    img.set_at((row,col), color)
    img.set_at((row,col+sz), color)
    img.set_at((row+sz,col-sz), color)
    img.set_at((row+sz,col), color)
    img.set_at((row+sz,col+sz), color)

def draw_path(img, path_x, path_y, color, shift_from_mid):
  for x, y in zip(path_x, path_y):
    draw_pt(img, x, y, color, shift_from_mid)

# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
  deg_to_rad = np.pi/180.
  slip_fator = 0.0014 # slip factor obtained from real data
  steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
  wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

  angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
  curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
  return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
  #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature

def draw_path_on(img, speed_ms, angle_steers, color=(0,0,255), shift_from_mid=0):
  path_x = np.arange(0., 50.1, 0.5)
  path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
  draw_path(img, path_x, path_y, color, shift_from_mid)

# ***** main loop *****
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Udacity SDC Challenge-2 Video viewer')
  parser.add_argument('--dataset', type=str, default="dataset.bag", help='Dataset/ROS Bag name')
  parser.add_argument('--skip', type=int, default="0", help='skip seconds')
  args = parser.parse_args()

  dataset = args.dataset
  skip = args.skip
  startsec = 0

  print "reading rosbag ", dataset
  bag = rosbag.Bag(dataset, 'r')
  for topic, msg, t in bag.read_messages(topics=['/center_camera/image_color','/right_camera/image_color','/left_camera/image_color','/vehicle/steering_report']):
    if startsec == 0:
        startsec = t.to_sec()
        if skip < 24*60*60:
            skipping = t.to_sec() + skip
            print "skipping ", skip, " seconds from ", startsec, " to ", skipping, " ..."
        else:
            skipping = skip
            print "skipping to ", skip, " from ", startsec, " ..."
    else:
        if t.to_sec() > skipping:
            if topic in ['/center_camera/image_color','/right_camera/image_color','/left_camera/image_color']:
                print(topic, msg.header.seq, t-msg.header.stamp, msg.height, msg.width, msg.encoding, t)
            else:
                print(topic, msg.header.seq, t-msg.header.stamp, msg.steering_wheel_angle, t)
                angle_steers = msg.steering_wheel_angle
        
            try: 
                if topic in ['/center_camera/image_color','/right_camera/image_color','/left_camera/image_color']:
                    # RGB_str = msg.data
                    RGB_str = np.fromstring(msg.data, dtype='uint8').reshape((640*480),3)[:, (2, 1, 0)].tostring() 
        
                    if topic == '/left_camera/image_color':
                        imgleft = pygame.transform.scale(pygame.image.fromstring(RGB_str, (640, 480), 'RGB'), (320, 240))
                    else:
                        if topic == '/center_camera/image_color':
                            imgcenter = pygame.transform.scale(pygame.image.fromstring(RGB_str, (640, 480), 'RGB'), (320, 240))
                        else:
                            imgright = pygame.transform.scale(pygame.image.fromstring(RGB_str, (640, 480), 'RGB'), (320, 240))
        
            except Exception, e:
               print("Error converting string to PyGame surface in StringToSurface", e)

            draw_path_on(imgleft, 0, angle_steers*20, (0,0,255), -10)
            draw_path_on(imgcenter, 0, angle_steers*20, (0,0,255), -20)
            draw_path_on(imgright, 0, angle_steers*20, (0,0,255), 0)

            screen.blit(imgleft, (0,0))
            screen.blit(imgcenter, (320,0))
            screen.blit(imgright, (640,0))

            pygame.display.flip()

