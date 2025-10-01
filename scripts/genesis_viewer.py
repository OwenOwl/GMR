import genesis as gs
import mujoco as mj
import numpy as np
from genesis.utils import geom as gu
from general_motion_retargeting.config.rb_config import RIGID_BODY_ID_MAP, RIGID_BODY_OFFSET, G1_TRACKED_LINK_NAMES

class GenesisViewer:
    def __init__(self, visualize=True):
        # Genesis initialization
        gs.init(backend=gs.gpu)

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=(1920, 1080),
                camera_pos=(3.5, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                refresh_rate=60,
                max_FPS=None,
            ),
            sim_options=gs.options.SimOptions(
                gravity=(0.0, 0.0, 0.0),
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
        self.rigid_body_offsets = {}
        self.target = {}

        # Only camera related
        self.world_rotation = np.eye(3)
        self.cameras = []
    
    def load_rigid_body_by_name(self, name, mode="Sphere", params={}, offset=None):
        rigid_body = None
        if mode == "Sphere":
            rigid_body = self.scene.add_entity(
                gs.morphs.Sphere(
                    radius=params.get("radius", 0.05),
                    pos=(0.0, 0.0, 0.0),
                    collision=False,
                    fixed=True,
                ),
                surface=gs.surfaces.Plastic(
                    color=params.get("color", (1.0, 1.0, 1.0)),
                ),
            )
        elif mode == "Box":
            rigid_body = self.scene.add_entity(
                gs.morphs.Box(
                    size=params.get("size", (0.1, 0.1, 0.1)),
                    pos=(0.0, 0.0, 0.0),
                    collision=False,
                    fixed=True,
                ),
                surface=gs.surfaces.Plastic(
                    color=params.get("color", (1.0, 1.0, 1.0)),
                ),
            )
        elif mode == "File":
            rigid_body = self.scene.add_entity(
                gs.morphs.Mesh(
                    file=params.get("path"),
                    scale=1.0,
                    pos=(0.0, 0.0, 0.0),
                    quat=(1.0, 0.0, 0.0, 0.0),
                    collision=False,
                    fixed=True,
                ),
                surface=gs.surfaces.Metal(
                    color=params.get("color", (1.0, 1.0, 1.0)),
                ),
            )
        elif mode == "Virtual": # No explicit build, only for calculation
            rigid_body = {
                "pos": np.array([0.0, 0.0, 0.0]),
                "quat": np.array([1.0, 0.0, 0.0, 0.0])
            }
        else:
            raise NotImplementedError(f"Rigid mode {mode} not implemented!")
        self.rigid_bodies[name] = rigid_body

        if offset is not None:
            self.rigid_body_offsets[name] = {
                "pos": np.array(offset["pos"]),
                "quat": np.array(offset["quat"]),
            }
        else:
            self.rigid_body_offsets[name] = {
                "pos": np.array([0.0, 0.0, 0.0]),
                "quat": np.array([1.0, 0.0, 0.0, 0.0]),
            }

    def initialize_cameras(self, camera_calibrations, rotation_matrix):
        self.world_rotation = np.array(rotation_matrix)
        for cam in camera_calibrations:
            pos = self.world_rotation @ np.array(cam["Position"])
            cam_wxyz = np.roll(np.array(cam["Orientation"]), 1)
            quat = gu.R_to_quat(self.world_rotation @ gu.quat_to_R(cam_wxyz))
            camera = self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.1, 0.1, 0.15),
                    pos=pos,
                    quat=quat,
                    collision=False,
                    fixed=True,
                ),
                surface=gs.surfaces.Plastic(
                    color=(0.5, 0.0, 1.0),
                ),
            )
            self.cameras.append(camera)
    
    def initialize_robot(self, mujoco_model=None):
        '''
        If mujoco_model is not None, the robot will always reorder the dofs according to the mujoco_model.
        '''
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/unitree_g1/g1_mocap_29dof.xml",
                pos=(-2, 0, 0),
                scale=1.0,
                collision=False,
            )
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
    
    def ik_dof_pos(self):
        if self.robot is None:
            return
        links = [self.robot.get_link(name) for name in G1_TRACKED_LINK_NAMES]
        poss = [self._get_pos_by_name(name) for name in G1_TRACKED_LINK_NAMES]
        quats = [self._get_quat_by_name(name) for name in G1_TRACKED_LINK_NAMES]
        qpos = self.robot.inverse_kinematics_multilink(
            links=links,
            poss=poss,
            quats=quats,
        )
        self.update_dof_pos(qpos)

    def update_rigid_body_by_name(self, name, pos, quat):
        if name not in self.rigid_bodies:
            raise ValueError(f"Rigid body {name} not found!")
        # aligned = pose * offset. Transform(v, u) = u * v
        aligned_pos = gu.transform_by_quat(
            self.rigid_body_offsets[name]["pos"], quat
        ) + pos
        aligned_quat = gu.transform_quat_by_quat(
            self.rigid_body_offsets[name]["quat"], quat
        )
        self._set_pos_by_name(name, aligned_pos)
        self._set_quat_by_name(name, aligned_quat)

    def update_rigid_bodies(self, frame):
        for name, (pos, quat) in frame.items():
            if name in self.rigid_bodies:
                self.update_rigid_body_by_name(name, pos, quat)

    def _get_pos_by_name(self, name):
        if name not in self.rigid_bodies:
            raise ValueError(f"Rigid body {name} not found!")
        if isinstance(self.rigid_bodies[name], gs.engine.entities.RigidEntity):
            return self.rigid_bodies[name].get_pos().cpu().numpy()
        elif isinstance(self.rigid_bodies[name], dict):
            return self.rigid_bodies[name]["pos"]
        else:
            raise NotImplementedError(f"Unknown type for rigid body {name}!")
    
    def _get_quat_by_name(self, name):
        if name not in self.rigid_bodies:
            raise ValueError(f"Rigid body {name} not found!")
        if isinstance(self.rigid_bodies[name], gs.engine.entities.RigidEntity):
            return self.rigid_bodies[name].get_quat().cpu().numpy()
        elif isinstance(self.rigid_bodies[name], dict):
            return self.rigid_bodies[name]["quat"]
        else:
            raise NotImplementedError(f"Unknown type for rigid body {name}!")

    def _set_pos_by_name(self, name, pos):
        if name not in self.rigid_bodies:
            raise ValueError(f"Rigid body {name} not found!")
        if isinstance(self.rigid_bodies[name], gs.engine.entities.RigidEntity):
            self.rigid_bodies[name].set_pos(pos)
        elif isinstance(self.rigid_bodies[name], dict):
            self.rigid_bodies[name]["pos"] = pos
        else:
            raise NotImplementedError(f"Unknown type for rigid body {name}!")

    def _set_quat_by_name(self, name, quat):
        if name not in self.rigid_bodies:
            raise ValueError(f"Rigid body {name} not found!")
        if isinstance(self.rigid_bodies[name], gs.engine.entities.RigidEntity):
            self.rigid_bodies[name].set_quat(quat)
        elif isinstance(self.rigid_bodies[name], dict):
            self.rigid_bodies[name]["quat"] = quat
        else:
            raise NotImplementedError(f"Unknown type for rigid body {name}!")

    def _get_offsets(self, frame):
        res = ""
        for name, (_, _) in frame.items():
            if name not in self.rigid_bodies or name not in self.target:
                continue
            target_pos = self.target[name].get_pos().cpu().numpy()
            target_quat = self.target[name].get_quat().cpu().numpy()
            pos = self.rigid_bodies[name].get_pos().cpu().numpy()
            quat = self.rigid_bodies[name].get_quat().cpu().numpy()
            # offset = pose^T * aligned. Transform(v, u) = u * v
            offset_pos = gu.transform_by_quat(
                target_pos - pos, gu.inv_quat(quat)
            )
            offset_quat = gu.transform_quat_by_quat(
                target_quat, gu.inv_quat(quat)
            )
            res += f'''\n"{name}": {{\n    "pos": {offset_pos.tolist()},\n    "quat": {offset_quat.tolist()},\n}},'''

        return res

    def step(self):
        self.scene.visualizer.update()
    
    ''' Setups '''
    
    def g1_links_setup_virtual(self):
        for name in G1_TRACKED_LINK_NAMES:
            self.load_rigid_body_by_name(
                name,
                mode="Virtual",
                offset=RIGID_BODY_OFFSET[name]
            )

    def g1_links_setup_offset(self):
        for name in G1_TRACKED_LINK_NAMES:
            self.load_rigid_body_by_name(
                name,
                mode="File",
                params={"path": f"xml/unitree_g1/meshes/{name}.STL", "color": (0.8, 0.0, 0.0)},
                offset=RIGID_BODY_OFFSET[name]
            )

    def MoCap_setup(self, args, mujoco_model=None):
        self.initialize_robot(mujoco_model=mujoco_model)
        self.build()
    
    def MoCap_step(self, qpos):
        self.update_dof_pos(qpos)
        self.step()
    
    def Real2Sim_setup(self, args):
        self.g1_links_setup_offset() # temp
        self.initialize_robot(mujoco_model=None)
        self.build()
    
    def Real2Sim_step(self, frame):
        self.update_rigid_bodies(frame)
        self.ik_dof_pos()
        self.step()
    
    def Real2Sim_offset_setup(self, args):
        self.g1_links_setup_offset()
        self.initialize_robot(mujoco_model=None)
        for name in G1_TRACKED_LINK_NAMES:
            self.target[name] = self.robot.get_link(name)
        self.build()

    def Real2Sim_offset_step(self, frame):
        self.update_rigid_bodies(frame)
        offset = self._get_offsets(frame)
        self.step()
        return offset

    ''' Only for testing / debugging purposes '''

    def test_block_setup(self, args):
        block_size = (0.36, 0.08, 0.04)
        self.load_rigid_body_by_name(
            "TestBlock1",
            mode="Box",
            params={"size": block_size, "color": (241./255, 198./255, 136./255)},
            offset=RIGID_BODY_OFFSET["TestBlock1"]
        )
        if args.get_offset:
            self.target["TestBlock1"] = self.scene.add_entity(
                gs.morphs.Box(
                    size=block_size,
                    pos=(0.0, 0.0, block_size[2] / 2),
                    collision=False,
                    fixed=True,
                )
            )