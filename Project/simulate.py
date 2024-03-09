from dm_control import mujoco
from dm_control import mjcf
import mujoco.viewer

import numpy as np
import time
import cv2

from model import *
import random

def find_nearest_food_pos(creature_pos, food_dict):
    dist_min = 10000
    pos_min = None
    for key, value in food_dict.items():
        if value['eaten'] == 1:
            continue
        food_pos = data.geom(value['name']).xpos
        dist = (creature_pos[0] - food_pos[0]) ** 2 + (creature_pos[1] - food_pos[1]) ** 2 + (creature_pos[2] - food_pos[2]) ** 2
        if dist < dist_min:
            dist_min = dist
            pos_min = value['pos']
    return pos_min

def simulate_render(model, data, creature_dict, food_dict, timestep, duration):
    viewer_handle = mujoco.viewer.launch_passive(model, data)
    renderer = mujoco.Renderer(model)

    frames = []
    mujoco.mj_resetData(model, data)

    for _ in range(int(duration/timestep)):
        viewer_handle.cam.azimuth = 45
        viewer_handle.cam.elevation = -45
        viewer_handle.cam.distance = 8.0
        viewer_handle.cam.lookat[:] = [0, 0, 0]

        viewer_handle.sync()

        for key, value in creature_dict.items():
            # print(value['name'])
            creature_pos = np.array(data.geom(value['name']).xpos)
            nearest_food_pos = np.array(find_nearest_food_pos(creature_pos, food_dict))
            if nearest_food_pos is not None:
                direction = nearest_food_pos - creature_pos
            else:
                direction = np.array([0, 0, 0])
            direction_norm = direction / np.linalg.norm(direction)
            data.joint(value['joint_name']).qvel = np.concatenate((0.1 * direction_norm * value['genotype'][0], [0, 0, 0]))
            # print(key)
            # print(creature_pos)
            # print(nearest_food_pos)
            # print(data.joint(value['joint_name']).qvel)

        mujoco.mj_step(model, data)

        for contact in data.contact.geom:
            if 'food' in data.geom(contact[0]).name and 'creature' in data.geom(contact[1]).name:
                # Remove food, Keep creature
                creature_name = data.geom(contact[1]).name.split('/')[-1]
                creature_dict[creature_name]['energy'] += 1
                food_name = data.geom(contact[0]).name.split('/')[-1]
                food_dict[food_name]['eaten'] = 1
            elif 'food' in data.geom(contact[1]).name and 'creature' in data.geom(contact[0]).name:
                # Remove food, Keep creature
                creature_name = data.geom(contact[0]).name.split('/')[-1]
                creature_dict[creature_name]['energy'] += 1
                food_name = data.geom(contact[1]).name.split('/')[-1]
                food_dict[food_name]['eaten'] = 1
            else:
                pass

        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels)
        time.sleep(timestep)
    viewer_handle.close()

    # frames = np.array(frames)
    # frame_count, height, width, channel = frames.shape
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter("./out.mp4", fourcc, framerate, (width, height))
    # for i in range(frame_count):
    #     out.write(frames[i])
    # out.release()

arena = mjcf.RootElement()
arena.worldbody.add('geom', type='plane', size=[2, 2, .1])
arena.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1])

genotype = []
TIMESTEP = 0.05
DURATION = 5
ITERATION = DURATION/TIMESTEP
creature_pos = (0, 0, 0.5)
genotype = [4, (0.3, 0.5, 0.7, 0.9)]
food_dict = {}
creature_dict = {}
for iter in range(10):
    food_count = random.randint(1, 4)
    food_site = arena.worldbody.add('site', pos=(0, 0, 0), group=2)
    for i in range(food_count):
        food_name = 'food_{:d}'.format(len(food_dict))
        food = Food(food_name, (-2, 2), (0.9, 0.1, 0.1, 0.9))
        food_model = food.make_model()
        food_site.attach(food_model)
        food_dict[food_name] = {'name' : '', 'geom_id' : -1, 'pos' : food.pos, 'eaten' : 0}

    food_to_remove = []
    for key, value in food_dict.items():
        if value['eaten'] == 1:
            food = arena.worldbody.find('geom', value['name'])
            food.remove()
            food_to_remove.append(key)
    print("Remove {:d} Foods".format(len(food_to_remove)))
    for food in food_to_remove:
        food_dict.pop(food, None)

    creature_pos = (creature_pos[0] + random.uniform(-1, 1), creature_pos[1] + random.uniform(-1, 1), creature_pos[2])
    genotype = [genotype[0] + random.randint(-1, 1), genotype[1]]
    creature_name = 'creature_{:d}'.format(len(creature_dict))
    creature = Creature(creature_name, genotype).make_model()
    creature_site = arena.worldbody.add('site', pos=creature_pos, group=3)
    creature_site.attach(creature).add('freejoint', name='{:s}_joint'.format(creature_name))
    creature_dict[creature_name] = {'name' : '', 'geom_id' : -1, 'genotype' : genotype, 'joint_name' : '', 'energy' : 1}
    
    model = mujoco.MjModel.from_xml_string(arena.to_xml_string())
    mujoco.mj_printModel(model, 'temp.txt')
    geom_id = 0
    while True:
        try:
            name = model.geom(geom_id).name
            key_name = name.split('/')[-1]
            if key_name in food_dict:
                food_dict[key_name]['geom_id'] = geom_id
                food_dict[key_name]['name'] = name
            if key_name in creature_dict:
                creature_dict[key_name]['geom_id'] = geom_id
                creature_dict[key_name]['name'] = name
        except:
            break
        geom_id += 1

    joint_id = 0
    while True:
        try:
            name = model.joint(joint_id).name
            key_name = name.split('/')[-2][0:-6]
            if key_name in creature_dict:
                creature_dict[key_name]['joint_name'] = name
        except:
            break
        joint_id += 1

    # print(creature_dict)
    data = mujoco.MjData(model)
    model.opt.gravity = (0, 0, 0)
    model.opt.timestep = TIMESTEP
    simulate_render(model, data, creature_dict, food_dict, TIMESTEP, DURATION)
    for creature in creature_dict.values():
        creature['energy'] -= 1

    creature_to_remove = []
    for key, value in creature_dict.items():
        if value['energy'] <= 0:
            creature = arena.worldbody.find('geom', value['name'])
            creature.rgba = (0.8, 0.8, 0.8, 0.9)
            # creature.remove()
            creature_to_remove.append(key)
    print("Remove {:d} Creatures".format(len(creature_to_remove)))
    for creature in creature_to_remove:
        creature_dict.pop(creature, None)

    print(food_dict)
    print(creature_dict)
