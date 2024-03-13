from dm_control import mujoco
from dm_control import mjcf
import mujoco.viewer

import numpy as np
import time
import cv2

from model import *
import random

def write_video(file_name, frames, fps):
    frames = np.array(frames)
    frames_color = np.zeros_like(frames)
    frame_count, height, width, channel = frames.shape
    for frame_index in range(frame_count):
        cv2.cvtColor(frames[frame_index], cv2.COLOR_RGB2BGR, frames_color[frame_index])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_name, fourcc, fps, (width, height))
    for i in range(frame_count):
        out.write(frames_color[i])
    out.release()

def find_nearest_food_pos(data, creature_pos, food_dict):
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

def simulate(model, data, creature_dict, food_dict, timestep, duration, frames, display):
    if display:
        viewer_handle = mujoco.viewer.launch_passive(model, data)
    renderer = mujoco.Renderer(model)
    mujoco.mj_resetData(model, data)

    # Simulation Loop
    for _ in range(int(duration/timestep)):
        if display:
            # Set up camera
            viewer_handle.cam.azimuth = 45
            viewer_handle.cam.elevation = -45
            viewer_handle.cam.distance = 8.0
            viewer_handle.cam.lookat[:] = [0, 0, 0]
            viewer_handle.sync()

        # Set movement for creatures
        for key, value in creature_dict.items():
            if value['alive'] == 1:
                creature_pos = np.array(data.geom(value['name']).xpos)
                nearest_food_pos = find_nearest_food_pos(data, creature_pos, food_dict)
                if nearest_food_pos is None:
                    direction_norm = np.array([0, 0, 0])
                else:
                    nearest_food_pos = np.array(nearest_food_pos)
                    direction = nearest_food_pos - creature_pos
                    direction_norm = direction / np.linalg.norm(direction)
                data.joint(value['joint_name']).qvel = np.concatenate((0.1 * direction_norm * value['genotype'][0], [0, 0, 0]))

        # Execute one step
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels)

        # Detect Collision and Eat Food
        for contact in data.contact.geom:
            if 'food' in data.geom(contact[0]).name and 'creature' in data.geom(contact[1]).name:
                # Remove food, Keep creature
                food_name = data.geom(contact[0]).name.split('/')[-1]
                creature_name = data.geom(contact[1]).name.split('/')[-1]
                if food_dict[food_name]['eaten'] == 0 and creature_dict[creature_name]['alive'] == 1:
                    food_dict[food_name]['eaten'] = 1
                    creature_dict[creature_name]['energy'] += 1
            elif 'creature' in data.geom(contact[0]).name and 'food' in data.geom(contact[1]).name:
                # Remove food, Keep creature
                food_name = data.geom(contact[1]).name.split('/')[-1]
                creature_name = data.geom(contact[0]).name.split('/')[-1]
                if food_dict[food_name]['eaten'] == 0 and creature_dict[creature_name]['alive'] == 1:
                    food_dict[food_name]['eaten'] = 1
                    creature_dict[creature_name]['energy'] += 1
            else:
                pass

        if display:
            time.sleep(timestep)

    if display:
        viewer_handle.close()
        time.sleep(0.1)

def iteration(arena, iterations, timestep, duration, food_range, frames, display):
    food_dict = {}
    creature_dict = {}
    creature_pos = (0, 0, 1)
    genotype = [4, (0.3, 0.5, 0.7, 0.9)]
    food_site = arena.worldbody.add('site', name='food_site', pos=(0, 0, 0), group=2)
    # Iteration Loop
    for iter in range(iterations):
        print("Iteration {:02d}".format(iter))
        # Add random food
        food_count = random.randint(food_range[0], food_range[1])
        for _ in range(food_count):
            food_name = 'food_{:d}'.format(len(food_dict))
            food_pos = (random.uniform(-2, 2), random.uniform(-2, 2), 0.1)
            food = Food(food_name, (-2, 2), (0.9, 0.1, 0.1, 0.9), food_pos)
            food_dict[food_name] = {'name' : '', 'geom_id' : -1, 'pos' : food_pos, 'eaten' : 0, 'hidden' : 0}
            food_model = food.make_model()
            food_site.attach(food_model)
        print("{:d} New Foods Created".format(food_count))

        # Creature reproduction
        creature_count = 0
        if iter == 0:
            creature_pos = (0, 0, 1)
            genotype = [4, (0.3, 0.5, 0.7, 0.9)]
            creature_name = 'creature_0'
            creature = Creature(creature_name, genotype).make_model()
            site_name = 'c{:s}_site'.format(creature_name.split('_')[1])
            creature_site = arena.worldbody.add('site', name=site_name, pos=creature_pos, group=3)
            creature_site.attach(creature).add('freejoint', name='{:s}_joint'.format(creature_name))
            creature_dict[creature_name] = {'name' : '', 'geom_id' : -1, 'genotype' : genotype, 'pos' : creature_pos, 'joint_name' : '', 'site_name' : site_name, 'energy' : 0.25*genotype[0], 'alive' : 1, 'aff_list' : []}
            creature_count += 1
        new_creature_dict = {}
        for key, value in creature_dict.items():
            if value['alive'] == 1:
                value['energy'] -= 0.25*value['genotype'][0]
                if value['energy'] >= 0.25*value['genotype'][0]:
                    # Have sufficient energy, reproduction
                    x_displace = random.uniform(0.3, 0.6) * (random.randint(0, 1)*2-1)
                    y_displace = random.uniform(0.3, 0.6) * (random.randint(0, 1)*2-1)
                    creature_pos = (value['pos'][0] + x_displace, value['pos'][1] + y_displace, value['pos'][2])
                    # creature_pos = (value['pos'][0] + random.uniform(-1, 1), value['pos'][1] + random.uniform(-1, 1), value['pos'][2])
                    genotype = [value['genotype'][0] + random.randint(-1, 1), value['genotype'][1]]
                    if genotype[0] < 2:
                        genotype[0] = 2
                    elif genotype[0] > 10:
                        genotype[0] = 10
                    value['energy'] -= 0.25*genotype[0]
                    creature_name = 'creature_{:d}'.format(len(creature_dict) + len(new_creature_dict))
                    creature = Creature(creature_name, genotype).make_model()
                    site_name = 'c{:s}_site'.format(creature_name.split('_')[1])
                    creature_site = arena.worldbody.add('site', name=site_name, pos=creature_pos, group=3)
                    creature_site.attach(creature).add('freejoint', name='{:s}_joint'.format(creature_name))
                    new_creature_dict[creature_name] = {'name' : '', 'geom_id' : -1, 'genotype' : genotype, 'pos' : creature_pos, 'joint_name' : '', 'site_name' : site_name, 'energy' : 0.25*genotype[0], 'alive' : 1, 'aff_list' : []}
                    creature_count += 1
        creature_dict.update(new_creature_dict)
        print("{:d} New Creatures Reproduced".format(creature_count))

        model = mujoco.MjModel.from_xml_string(arena.to_xml_string())
        mujoco.mj_printModel(model, './Project/model.txt')
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
                if 'thigh' in key_name:
                    creature_dict["creature_{:s}".format(key_name.split('_')[0][1:])]['aff_list'].append(name)
                if 'hip' in key_name:
                    creature_dict["creature_{:s}".format(key_name.split('_')[0][1:])]['aff_list'].append(name)
                if 'neck' in key_name:
                    creature_dict["creature_{:s}".format(key_name.split('_')[0][1:])]['aff_list'].append(name)
                if 'head' in key_name:
                    creature_dict["creature_{:s}".format(key_name.split('_')[0][1:])]['aff_list'].append(name)
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

        data = mujoco.MjData(model)
        model.opt.gravity = (0, 0, 0)
        model.opt.timestep = timestep
        simulate(model, data, creature_dict, food_dict, timestep, duration, frames, display)

        # Delete Eaten Food
        removed_food = 0
        total_foods = 0
        for key, value in food_dict.items():
            if value['eaten'] == 1 and value['hidden'] == 0:
                food = arena.worldbody.find('geom', value['name'])
                if food is not None:
                    food.remove()
                    value['hidden'] = 1
                    removed_food += 1
            elif value['eaten'] == 0:
                total_foods += 1
        print("{:d} Foods eaten".format(removed_food))
        print("{:d} Foods left".format(total_foods))

        # Delete Dead Creature
        removed_creature = 0
        total_creatures = 0
        for key, value in creature_dict.items():
            if value['alive'] == 1:
                if value['energy'] <= 0:
                    value['alive'] = 0
                    creature = arena.worldbody.find('geom', value['name'])
                    creature.rgba = (0.8, 0.8, 0.8, 0.9)
                    # creature.remove()
                    for aff_name in value['aff_list']:
                        creature_aff = arena.worldbody.find('geom', aff_name)
                        creature_aff.rgba = (0.8, 0.8, 0.8, 0.9)
                        # creature_aff.remove()
                    creature_site = arena.worldbody.find('site', value['site_name'])
                    creature_site.remove()
                    # print(value['joint_name'])
                    # creature_joint = arena.worldbody.find('joint', value['joint_name'])
                    # creature_joint.remove()
                    # creature.remove()
                    removed_creature += 1
                elif value['energy'] > 0:
                    # creature_pos = data.geom(value['name']).xpos
                    # creature_site = arena.worldbody.find('site', value['site_name'])
                    # creature_site.pos = creature_pos
                    total_creatures += 1
        print("{:d} Creatures died".format(removed_creature))
        print("{:d} Creatures survived".format(total_creatures))

        # print(food_dict)
        # print(creature_dict)

def iteration_random(arena, iterations, timestep, duration, food_range, frames, display):
    food_dict = {}
    creature_dict = {}
    food_site = arena.worldbody.add('site', name='food_site', pos=(0, 0, 0), group=2)
    # Iteration Loop
    for iter in range(iterations):
        print("Iteration {:02d}".format(iter))
        # Add random food
        food_count = random.randint(food_range[0], food_range[1])
        for _ in range(food_count):
            food_name = 'food_{:d}'.format(len(food_dict))
            food_pos = (random.uniform(-2, 2), random.uniform(-2, 2), 0.1)
            food = Food(food_name, (-2, 2), (0.9, 0.1, 0.1, 0.9), food_pos)
            food_dict[food_name] = {'name' : '', 'geom_id' : -1, 'pos' : food_pos, 'eaten' : 0, 'hidden' : 0}
            food_model = food.make_model()
            food_site.attach(food_model)
        for key, value in creature_dict.items():
            if value['alive'] == 1:
                value['energy'] -= 0.25*value['genotype'][0]

        print("{:d} New Foods Created".format(food_count))

        # Creature reproduction
        creature_count = random.randint(1, 2)
        for i in range(creature_count):
            creature_pos = (random.uniform(-2, 2), random.uniform(-2, 2), 1)
            genotype = [random.randint(2, 10), (0.3, 0.5, 0.7, 0.9)]
            creature_name = 'creature_{:d}'.format(len(creature_dict))
            creature = Creature(creature_name, genotype).make_model()
            site_name = 'c{:s}_site'.format(creature_name.split('_')[1])
            creature_site = arena.worldbody.add('site', name=site_name, pos=creature_pos, group=3)
            creature_site.attach(creature).add('freejoint', name='{:s}_joint'.format(creature_name))
            creature_dict[creature_name] = {'name' : '', 'geom_id' : -1, 'genotype' : genotype, 'pos' : creature_pos, 'joint_name' : '', 'site_name' : site_name, 'energy' : 0.25*genotype[0], 'alive' : 1, 'aff_list' : []}
        print("{:d} New Creatures Reproduced".format(creature_count))

        model = mujoco.MjModel.from_xml_string(arena.to_xml_string())
        mujoco.mj_printModel(model, './Project/model.txt')
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
                if 'thigh' in key_name:
                    creature_dict["creature_{:s}".format(key_name.split('_')[0][1:])]['aff_list'].append(name)
                if 'hip' in key_name:
                    creature_dict["creature_{:s}".format(key_name.split('_')[0][1:])]['aff_list'].append(name)
                if 'neck' in key_name:
                    creature_dict["creature_{:s}".format(key_name.split('_')[0][1:])]['aff_list'].append(name)
                if 'head' in key_name:
                    creature_dict["creature_{:s}".format(key_name.split('_')[0][1:])]['aff_list'].append(name)
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

        data = mujoco.MjData(model)
        model.opt.gravity = (0, 0, 0)
        model.opt.timestep = timestep
        simulate(model, data, creature_dict, food_dict, timestep, duration, frames, display)

        # Delete Eaten Food
        removed_food = 0
        total_foods = 0
        for key, value in food_dict.items():
            if value['eaten'] == 1 and value['hidden'] == 0:
                food = arena.worldbody.find('geom', value['name'])
                if food is not None:
                    food.remove()
                    value['hidden'] = 1
                    removed_food += 1
            elif value['eaten'] == 0:
                total_foods += 1
        print("{:d} Foods eaten".format(removed_food))
        print("{:d} Foods left".format(total_foods))

        # Delete Dead Creature
        removed_creature = 0
        total_creatures = 0
        for key, value in creature_dict.items():
            if value['alive'] == 1:
                if value['energy'] <= 0:
                    value['alive'] = 0
                    creature = arena.worldbody.find('geom', value['name'])
                    creature.rgba = (0.8, 0.8, 0.8, 0.9)
                    # creature.remove()
                    for aff_name in value['aff_list']:
                        creature_aff = arena.worldbody.find('geom', aff_name)
                        creature_aff.rgba = (0.8, 0.8, 0.8, 0.9)
                        # creature_aff.remove()
                    creature_site = arena.worldbody.find('site', value['site_name'])
                    creature_site.remove()
                    # print(value['joint_name'])
                    # creature_joint = arena.worldbody.find('joint', value['joint_name'])
                    # creature_joint.remove()
                    # creature.remove()
                    removed_creature += 1
                elif value['energy'] > 0:
                    # creature_pos = data.geom(value['name']).xpos
                    # creature_site = arena.worldbody.find('site', value['site_name'])
                    # creature_site.pos = creature_pos
                    total_creatures += 1
        print("{:d} Creatures died".format(removed_creature))
        print("{:d} Creatures survived".format(total_creatures))

        # print(food_dict)
        # print(creature_dict)
        
arena = mjcf.RootElement()
arena.worldbody.add('geom', type='plane', size=[2, 2, .1])
arena.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1])

# frames = []
# iteration_random(arena, 15, 0.05, 8, (2, 4), frames, False)
# write_video("./project_random.mp4", frames, 20)

frames = []
iteration(arena, 15, 0.05, 8, (2, 4), frames, False)
write_video("./project_evolve.mp4", frames, 20)