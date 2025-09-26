import genesis as gs
import mujoco as mj
import numpy as np
from genesis.utils import geom as gu

class GenesisViewer:
    def __init__(self):
        # Genesis initialization
        gs.init(backend=gs.gpu)

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=True,
        )

        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.robot = None
        self.robot_dofs = None

        self.world_rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        self.cameras = []
    
    def set_world_rotation(self, rotation_matrix):
        self.world_rotation = np.array(rotation_matrix)
    
    def initialize_cameras(self, camera_calibrations):
        for cam in camera_calibrations:
            pos = self.world_rotation @ np.array(cam["Position"])
            cam_wxyz = np.array(cam["Orientation"][3:] + cam["Orientation"][:3])
            quat = gu.R_to_quat(self.world_rotation @ gu.quat_to_R(cam_wxyz))
            camera = self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.1, 0.1, 0.3),
                    pos=pos,
                    quat=quat,
                    collision=False,
                    fixed=True,
                )
            )
            self.cameras.append(camera)
    
    def initialize_robot(self, mujoco_model=None):
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/unitree_g1/g1_mocap_29dof.xml", scale=1.0, collision=True)
        )
        if mujoco_model is None:
            self.robot_dofs = list(range(6, 35))
        else:
            dof_names = [
                mj.mj_id2name(mujoco_model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(mujoco_model.njnt)
            ]
            self.robot_dofs = [
                self.robot.get_joint(name).dofs_idx_local[0] for name in dof_names[1:]
            ]
    
    def build(self):
        self.scene.build()
    
    def update_dof_pos(self, dof_pos):
        if self.robot is None:
            return
        self.robot.set_pos(dof_pos[:3])
        self.robot.set_quat(dof_pos[3:7])
        self.robot.set_dofs_position(dof_pos[7:], dofs_idx_local=self.robot_dofs)
    
    def step(self, dof_pos=None):
        self.scene.step()
