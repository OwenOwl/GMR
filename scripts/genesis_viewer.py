import genesis as gs
import mujoco as mj
import numpy as np
from genesis.utils import geom as gu

class GenesisViewer:
    def __init__(self, visualize=True):
        # Genesis initialization
        gs.init(backend=gs.gpu)

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=600,
            ),
            show_viewer=visualize,
            show_FPS=False,
        )

        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.robot = None
        self.robot_dofs = None

        self.rigid_bodies = {}

        self.world_rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        self.cameras = []
    
    def set_world_rotation(self, rotation_matrix):
        self.world_rotation = np.array(rotation_matrix)
    
    def initialize_rigid_body_by_name(self, name, mode="Sphere", asset=None):
        if mode == "Sphere":
            rigid_body = self.scene.add_entity(
                gs.morphs.Sphere(
                    radius=0.05,
                    pos=(0.0, 0.0, 0.0),
                    collision=False,
                    fixed=True,
                )
            )
        elif mode == "Test_Block":
            rigid_body = self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.3, 0.1, 0.1),
                    pos=(0.0, 0.0, 0.0),
                    collision=False,
                    fixed=True,
                )
            )
        else:
            raise NotImplementedError(f"Rigid mode {mode} not implemented!")
        self.rigid_bodies[name] = rigid_body

    def initialize_cameras(self, camera_calibrations):
        for cam in camera_calibrations:
            pos = self.world_rotation @ np.array(cam["Position"])
            cam_wxyz = np.roll(np.array(cam["Orientation"]), 1)
            quat = gu.R_to_quat(self.world_rotation @ gu.quat_to_R(cam_wxyz))
            camera = self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.1, 0.1, 0.2),
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
    
    def update_rigid_body_by_name(self, name, pos, quat):
        if name not in self.rigid_bodies:
            print(f"Rigid body {name} not found!")
            return
        self.rigid_bodies[name].set_pos(pos)
        self.rigid_bodies[name].set_quat(quat)
    
    def update_rigid_bodies(self, frame):
        for name, (pos, quat) in frame.items():
            if name in self.rigid_bodies:
                self.update_rigid_body_by_name(name, pos, quat)

    def step(self, dof_pos=None):
        self.scene.step()
