
from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from genesis_viewer import GenesisViewer
from config import Camera_Calibrations, World_Rotation
import threading
import argparse

def main(args):
    genesis_env = GenesisViewer(visualize=True)
    genesis_env.set_world_rotation(World_Rotation)
    genesis_env.initialize_cameras(Camera_Calibrations)

    genesis_env.initialize_rigid_body_by_name("TestBlock1", mode="Test_Block")

    genesis_env.build()

    client = setup_optitrack(
        server_address=args.server_ip,
        client_address=args.client_ip,
        use_multicast=args.use_multicast,
    )

    # start a thread to client.run()
    thread = threading.Thread(target=client.run)
    thread.start()

    if not client:
        print("Failed to setup OptiTrack client")
        exit(1)

    while True:
        frame = client.get_frame()
        frame_number = client.get_frame_number()

        genesis_env.update_rigid_bodies(frame)

        genesis_env.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="192.168.0.232")
    parser.add_argument("--client_ip", type=str, default="192.168.0.128")
    parser.add_argument("--use_multicast", type=bool, default=False)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    args = parser.parse_args()
    main(args)
    