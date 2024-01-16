import dm_control.mujoco 
import mujoco.viewer
import time

m = dm_control.mujoco.MjModel.from_xml_path("example.xml")
d = dm_control.mujoco.MjData(m)
viewer_handle = mujoco.viewer.launch_passive(m, d)
body1_jntadr = m.body_jntadr[1]  # Assuming body1 is the second body in the XML
body1_qveladr = body1_jntadr + 3  # For free joints, the first 3 elements are position, next 3 are velocity

for i in range(2000):
    # Set camera parameters
    # These parameters can be adjusted to change the camera angle and perspective
    viewer_handle.cam.azimuth = 0  # Azimuthal angle (in degrees)
    viewer_handle.cam.elevation = -20  # Elevation angle (in degrees)
    viewer_handle.cam.distance = 15.0  # Distance from the camera to the target
    viewer_handle.cam.lookat[:] = [0.0, 0.0, 0.0]  # X-coordinate of the target position
    d.qvel[body1_qveladr] = 1.0
    viewer_handle.sync()
    dm_control.mujoco.mj_step(m, d)
    time.sleep(0.01)
    