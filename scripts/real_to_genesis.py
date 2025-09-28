
from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from genesis_viewer import GenesisViewer
import threading
import argparse

def main(args):
    genesis_env = GenesisViewer(visualize=True)

    from general_motion_retargeting.config.camera_config import Camera_Calibrations, World_Rotation
    genesis_env.initialize_cameras(Camera_Calibrations, World_Rotation)

    genesis_env.test_setup(args)

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

    tic = 0
    while True:
        tic += 1
        frame = client.get_frame()
        frame_number = client.get_frame_number()

        if tic == 1:
            print(f"Data received! First frame number: {frame_number}")

        genesis_env.update_rigid_bodies(frame)

        if args.get_offset and tic % 100 == 0:
            for name, (_, _) in frame.items():
                offset_pos, offset_quat = genesis_env.get_offset(name)
                print(f'''\n"{name}": {{\n    "pos": {offset_pos.tolist()},\n    "quat": {offset_quat.tolist()},\n}}''')

        genesis_env.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="192.168.0.232")
    parser.add_argument("--client_ip", type=str, default="192.168.0.128")
    parser.add_argument("--use_multicast", type=bool, default=False)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--get_offset", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
    