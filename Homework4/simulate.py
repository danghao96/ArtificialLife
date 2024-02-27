from model import *
import random

def simulate(model, data, timestep, duration):
    viewer_handle = mujoco.viewer.launch_passive(model, data)
    renderer = mujoco.Renderer(model)

    frames = []
    mujoco.mj_resetData(model, data)
    for _ in range(int(duration/timestep)):
        print('Generalized positions:', data.qpos)
        # print('Generalized velocities:', d.qvel)
        viewer_handle.sync()
        mujoco.mj_step(model, data)
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

TIMESTEP = 0.05
DURATION = 5
ITERATION = DURATION/TIMESTEP
creature_pos = (0, 0, 0.5)
for _ in range(5):
    creature = Creature(num_legs=4, rgba=(0.3, 0.5, 0.7, 0.9)).make_model()
    creature_pos = (creature_pos[0] + random.uniform(-1, 1), creature_pos[1] + random.uniform(-1, 1), creature_pos[2])
    creature_site = arena.worldbody.add('site', pos=creature_pos, group=3)
    creature_site.attach(creature).add('freejoint')
    model = mujoco.MjModel.from_xml_string(arena.to_xml_string())
    data = mujoco.MjData(model)
    model.opt.gravity = (0, 0, -10)
    model.opt.timestep = TIMESTEP
    simulate(model, data, TIMESTEP, DURATION)
