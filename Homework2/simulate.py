from dm_control import mujoco
from dm_control import mjcf
import mujoco.viewer

import numpy as np
import time
import cv2

def create_limb(length, rgba):
    model = mjcf.RootElement()

    model.default.joint.damping = 2
    model.default.joint.type = 'hinge'
    model.default.geom.type = 'capsule'
    model.default.geom.rgba = rgba

    # Thigh:
    thigh = model.worldbody.add('body')
    hip = thigh.add('joint', axis=[0, 0, 1])
    thigh.add('geom', fromto=[0, 0, 0, length, 0, 0], size=[length/4])

    # Hip:
    shin = thigh.add('body', pos=[length, 0, 0])
    knee = shin.add('joint', axis=[0, 1, 0])
    shin.add('geom', fromto=[0, 0, 0, 0, 0, -length], size=[length/5])

    # Position actuators:
    model.actuator.add('position', joint=hip, kp=10)
    model.actuator.add('position', joint=knee, kp=10)
    
    return model

def create_head(length, rgba):
    model = mjcf.RootElement()

    model.default.joint.damping = 2
    model.default.joint.type = 'hinge'
    model.default.geom.type = 'capsule'
    model.default.geom.rgba = rgba

    # Neck:
    neck = model.worldbody.add('body')
    vertebrae = neck.add('joint', axis=[0, 0, 1])
    neck.add('geom', fromto=[0, 0, 0, length, 0, 0], size=[length/4])

    # Head:
    head = model.worldbody.add('body', pos=[-length, 0, 0])
    atlas = head.add('joint', axis=[0, 0, 1])
    head.add('geom', type='sphere', size=[length])

    # Position actuators:
    model.actuator.add('position', joint=vertebrae, kp=10)
    model.actuator.add('position', joint=atlas, kp=10)
    
    return model

def make_creature(num_legs):
    BODY_RADIUS = 0.1
    BODY_SIZE = (BODY_RADIUS, BODY_RADIUS, BODY_RADIUS / 2)

    rgba = (0.3, 0.5, 0.7, 0.9)
    model = mjcf.RootElement()
    model.compiler.angle = 'radian'
    model.worldbody.add('geom', name='torso', type='ellipsoid', size=BODY_SIZE, rgba=rgba)

    for i in range(num_legs):
        theta = 2 * i * np.pi / num_legs
        hip_pos = BODY_RADIUS * np.array([np.cos(theta), np.sin(theta), 0])
        hip_site = model.worldbody.add('site', pos=hip_pos, euler=[0, 0, theta])
        leg = create_limb(length=BODY_RADIUS, rgba=rgba)
        hip_site.attach(leg)

    head_pos = BODY_RADIUS * np.array([0, 0, 1])
    head_site = model.worldbody.add('site', pos=head_pos, euler=[0, np.pi/2, 0])
    head = create_head(length=BODY_RADIUS/2, rgba=rgba)
    head_site.attach(head)

    return model

arena = mjcf.RootElement()
arena.worldbody.add('geom', type='plane', size=[2, 2, .1])
arena.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1])

creature = make_creature(num_legs=4)
creature_pos = (0, 0, 0.1)
creature_site = arena.worldbody.add('site', pos=creature_pos, group=3)
creature_site.attach(creature).add('freejoint')

model = mujoco.MjModel.from_xml_string(arena.to_xml_string())
data = mujoco.MjData(model)
model.opt.gravity = (0, 0, -10)
viewer_handle = mujoco.viewer.launch_passive(model, data)
renderer = mujoco.Renderer(model)

frames = []
mujoco.mj_resetData(model, data)
for i in range(120):
    print('Generalized positions:', data.qpos)
    # print('Generalized velocities:', d.qvel)
    viewer_handle.sync()
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()
    frames.append(pixels)
    time.sleep(0.03)
    
duration = 4
framerate = 30

frames = np.array(frames)
frame_count, height, width, channel = frames.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("./out.mp4", fourcc, framerate, (width, height))
for i in range(frame_count):
    out.write(frames[i])
out.release()