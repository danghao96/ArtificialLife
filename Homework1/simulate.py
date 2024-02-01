from dm_control import mujoco
import mujoco.viewer
import time
import cv2
import numpy as np

m = mujoco.MjModel.from_xml_path("example.xml")
d = mujoco.MjData(m)
m.opt.gravity = (0, 0, -10)
viewer_handle = mujoco.viewer.launch_passive(m, d)
renderer = mujoco.Renderer(m)

duration = 8  # (seconds)
framerate = 60  # (Hz)
frames = []
mujoco.mj_resetData(m, d)
for i in range(6000):
    # Set camera parameters
    # These parameters can be adjusted to change the camera angle and perspective
    viewer_handle.cam.azimuth = 0  # Azimuthal angle (in degrees)
    viewer_handle.cam.elevation = -20  # Elevation angle (in degrees)
    viewer_handle.cam.distance = 10.0  # Distance from the camera to the target
    viewer_handle.cam.lookat[:] = d.qpos[:3]  # X-coordinate of the target position

    car_id = m.geom('car_geom')
    d.qvel[0] = 10

    print('Generalized positions:', d.qpos)
    # print('Generalized velocities:', d.qvel)
    viewer_handle.sync()
    mujoco.mj_step(m, d)
    renderer.update_scene(d)
    pixels = renderer.render()
    frames.append(pixels)
    time.sleep(0.003)
    
# frames = np.array(frames)
# frame_count, height, width, channel = frames.shape
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("./out.mp4", fourcc, framerate, (width, height))
# for i in range(frame_count):
#     out.write(frames[i])
# out.release()