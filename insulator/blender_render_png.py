#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
blender /home/nolan/Desktop/renderfordarknet/blank.blend --background --python /home/nolan/Desktop/renderfordarknet/render_model_views.py

author: hao su, charles r. qi, yangyan li
'''

import os
import bpy
import sys
import math
import random
import numpy as np

# Load rendering light parameters
light_num_lowbound = 2
light_num_highbound = 5
light_dist_lowbound = 3
light_dist_highbound = 10

g_syn_light_azimuth_degree_lowbound = 0
g_syn_light_azimuth_degree_highbound = 360
g_syn_light_elevation_degree_lowbound = 0
g_syn_light_elevation_degree_highbound = 90
g_syn_light_energy_mean = 0.005
g_syn_light_energy_std = 0.01
g_syn_light_environment_energy_lowbound = 0
g_syn_light_environment_energy_highbound = 1


def camPosToQuaternion(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    axis = (-cz, 0, cx)
    angle = math.acos(cy)
    a = math.sqrt(2) / 2
    b = math.sqrt(2) / 2
    w1 = axis[0]
    w2 = axis[1]
    w3 = axis[2]
    c = math.cos(angle / 2)
    d = math.sin(angle / 2)
    q1 = a * c - b * d * w1
    q2 = b * c + a * d * w1
    q3 = a * d * w2 + b * d * w3
    q4 = -b * d * w2 + a * d * w3
    return (q1, q2, q3, q4)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist    
    t = math.sqrt(cx * cx + cy * cy) 
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    #roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll    
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta): 
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy): 
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e    
    return (q1, q2, q3, q4)

def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

# Input parameters
base_dir = os.path.dirname(os.path.abspath(__file__))
#syn_images_folder = os.path.join(base_dir, 'syn_images')
syn_images_folder = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/insulator_raw_images2'
if not os.path.exists(syn_images_folder):
    os.mkdir(syn_images_folder)
view_params = [[0, 45, 0, 2]]

CAD_path = os.path.join(base_dir, 'CAD/insulator.obj')
bpy.ops.import_scene.obj(filepath=CAD_path)
#bpy.ops.import_mesh.stl(filepath="/home/nolan/Desktop/insulator/Part1.STL")
bpy.ops.transform.resize(value=(0.01,0.01,0.01))
bpy.ops.transform.translate(value=(0,0,-0.35))


bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
#bpy.context.scene.render.use_shadows = False
#bpy.context.scene.render.use_raytrace = False

bpy.data.objects['Lamp'].data.energy = 0

#m.subsurface_scattering.use = True

camObj = bpy.data.objects['Camera']
# camObj.data.lens_unit = 'FOV'
# camObj.data.angle = 0.2
bpy.data.scenes["Scene"].render.resolution_x = 128*4
bpy.data.scenes["Scene"].render.resolution_y = 128*3
# set lights
bpy.ops.object.select_all(action='TOGGLE')
if 'Lamp' in list(bpy.data.objects.keys()):
    bpy.data.objects['Lamp'].select = True # remove default light
bpy.ops.object.delete()

# YOUR CODE START HERE
img_num = 0
elevation_deg_max = 60
#for param in view_params:
for i in range(30000):
    azimuth_deg = 0
    elevation_deg = (np.random.rand()-0.5) * 2 * elevation_deg_max
    theta_deg = np.random.rand() * 359
    rho = np.random.rand() * 5 + 2

    # clear default lights
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    # set environment lighting
    #bpy.context.space_data.context = 'WORLD'
    bpy.context.scene.world.light_settings.use_environment_light = True
    bpy.context.scene.world.light_settings.environment_energy = np.random.uniform(g_syn_light_environment_energy_lowbound, g_syn_light_environment_energy_highbound)
    bpy.context.scene.world.light_settings.environment_color = 'PLAIN'

    # set point lights
    for i in range(random.randint(light_num_lowbound,light_num_highbound)):
        light_azimuth_deg = np.random.uniform(g_syn_light_azimuth_degree_lowbound, g_syn_light_azimuth_degree_highbound)
        light_elevation_deg  = np.random.uniform(g_syn_light_elevation_degree_lowbound, g_syn_light_elevation_degree_highbound)
        light_dist = np.random.uniform(light_dist_lowbound, light_dist_highbound)
        lx, ly, lz = obj_centened_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
        bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
        bpy.data.objects['Point'].data.energy = np.random.normal(g_syn_light_energy_mean, g_syn_light_energy_std)

    cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
    q1 = camPosToQuaternion(cx, cy, cz)
    q2 = camRotQuaternion(cx, cy, cz, theta_deg)
    q = quaternionProduct(q2, q1)
    camObj.location[0] = cx
    camObj.location[1] = cy 
    camObj.location[2] = cz
    camObj.rotation_mode = 'QUATERNION'
    camObj.rotation_quaternion[0] = q[0]
    camObj.rotation_quaternion[1] = q[1]
    camObj.rotation_quaternion[2] = q[2]
    camObj.rotation_quaternion[3] = q[3]
    # ** multiply tilt by -1 to match pascal3d annotations **
    #theta_deg = (-1*theta_deg)%360
    img_num += 1
    label = int((elevation_deg + elevation_deg_max)/10)
    img_name = 'no%d_%d.png'%(img_num, label)
    bpy.data.scenes['Scene'].render.filepath = os.path.join(syn_images_folder, img_name)
    bpy.ops.render.render( write_still=True )
    
