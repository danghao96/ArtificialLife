from dm_control import mjcf
import numpy as np

class Creature:
    def __init__(self, name, genotype):
        self.num_legs = genotype[0]
        self.rgba = genotype[1]
        self.name = name

    def create_limb(self, length, rgba):
        model = mjcf.RootElement()

        model.default.joint.damping = 2
        model.default.joint.type = 'hinge'
        model.default.geom.type = 'capsule'
        model.default.geom.rgba = rgba

        # Thigh:
        thigh = model.worldbody.add('body')
        hip = thigh.add('joint', axis=[0, 0, 1])
        thigh.add('geom', name='c{:s}_thigh'.format(self.name.split('_')[1]), fromto=[0, 0, 0, length, 0, 0], size=[length/4])

        # Hip:
        shin = thigh.add('body', pos=[length, 0, 0])
        knee = shin.add('joint', axis=[0, 1, 0])
        shin.add('geom', name='c{:s}_hip'.format(self.name.split('_')[1]), fromto=[0, 0, 0, 0, 0, -length], size=[length/5])

        # Position actuators:
        # model.actuator.add('position', joint=hip, kp=10)
        # model.actuator.add('position', joint=knee, kp=10)
        
        return model

    def create_head(self, length, rgba):
        model = mjcf.RootElement()

        model.default.joint.damping = 2
        model.default.joint.type = 'hinge'
        model.default.geom.type = 'capsule'
        model.default.geom.rgba = rgba

        # Neck:
        neck = model.worldbody.add('body')
        vertebrae = neck.add('joint', axis=[0, 0, 1])
        neck.add('geom', name='c{:s}_neck'.format(self.name.split('_')[1]), fromto=[0, 0, 0, length, 0, 0], size=[length/4])

        # Head:
        head = model.worldbody.add('body', pos=[-length, 0, 0])
        atlas = head.add('joint', axis=[0, 0, 1])
        head.add('geom', name='c{:s}_head'.format(self.name.split('_')[1]), type='sphere', size=[length])

        # Position actuators:
        # model.actuator.add('position', joint=vertebrae, kp=10)
        # model.actuator.add('position', joint=atlas, kp=10)
        
        return model

    def make_model(self):
        BODY_RADIUS = 0.1
        BODY_SIZE = (BODY_RADIUS, BODY_RADIUS, BODY_RADIUS / 2)

        model = mjcf.RootElement()
        model.compiler.angle = 'radian'
        model.worldbody.add('geom', name=self.name, type='ellipsoid', size=BODY_SIZE, rgba=self.rgba)

        for i in range(self.num_legs):
            theta = 2 * i * np.pi / self.num_legs
            hip_pos = BODY_RADIUS * np.array([np.cos(theta), np.sin(theta), 0])
            hip_site = model.worldbody.add('site', pos=hip_pos, euler=[0, 0, theta])
            leg = self.create_limb(length=BODY_RADIUS, rgba=self.rgba)
            hip_site.attach(leg)

        head_pos = BODY_RADIUS * np.array([0, 0, 1])
        head_site = model.worldbody.add('site', pos=head_pos, euler=[0, np.pi/2, 0])
        head = self.create_head(length=BODY_RADIUS/2, rgba=self.rgba)
        head_site.attach(head)

        return model

class Food:
    def __init__(self, name, world_size, rgba, pos):
        self.name = name
        self.world_size = world_size
        self.rgba = rgba
        self.pos = pos
        
    def make_model(self):
        BODY_RADIUS = 0.1
        BODY_SIZE = (BODY_RADIUS/2, BODY_RADIUS/2, BODY_RADIUS/2)

        model = mjcf.RootElement()
        model.compiler.angle = 'radian'
        model.worldbody.add('geom', name=self.name, type='sphere', pos=self.pos, size=BODY_SIZE, rgba=self.rgba)
        return model