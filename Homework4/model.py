from dm_control import mujoco
from dm_control import mjcf
import mujoco.viewer

import numpy as np
import time
import cv2

class Creature:
    def __init__(self, num_legs, rgba):
        self.num_legs = num_legs
        self.rgba = rgba

    def make_model(self):
        BODY_RADIUS = 0.1
        BODY_SIZE = (BODY_RADIUS, BODY_RADIUS, BODY_RADIUS / 2)

        model = mjcf.RootElement()
        model.compiler.angle = 'radian'
        model.worldbody.add('geom', name='torso', type='ellipsoid', size=BODY_SIZE, rgba=self.rgba)

        for i in range(self.num_legs):
            theta = 2 * i * np.pi / self.num_legs
            hip_pos = BODY_RADIUS * np.array([np.cos(theta), np.sin(theta), 0])
            hip_site = model.worldbody.add('site', pos=hip_pos, euler=[0, 0, theta])
            leg = create_limb(length=BODY_RADIUS, rgba=self.rgba)
            hip_site.attach(leg)

        head_pos = BODY_RADIUS * np.array([0, 0, 1])
        head_site = model.worldbody.add('site', pos=head_pos, euler=[0, np.pi/2, 0])
        head = create_head(length=BODY_RADIUS/2, rgba=self.rgba)
        head_site.attach(head)

        return model


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
