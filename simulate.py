from dm_control import mujoco
import mujoco.viewer
import time

m = mujoco.MjModel.from_xml_path("example.xml")
d = mujoco.MjData(m)
m.opt.gravity = (0, 0, -10)

viewer_handle = mujoco.viewer.launch_passive(m, d)
for i in range(4000):
    # Set camera parameters
    # These parameters can be adjusted to change the camera angle and perspective
    viewer_handle.cam.azimuth = 0  # Azimuthal angle (in degrees)
    viewer_handle.cam.elevation = -20  # Elevation angle (in degrees)
    viewer_handle.cam.distance = 100.0  # Distance from the camera to the target
    viewer_handle.cam.lookat[:] = [0.0, 0.0, 0.0]  # X-coordinate of the target position

    car_id = m.geom('car_geom')
    d.qvel[0] = 10

    print('Generalized positions:', d.qpos)
    print('Generalized velocities:', d.qvel)
    viewer_handle.sync()
    mujoco.mj_step(m, d)
    time.sleep(0.01)
    