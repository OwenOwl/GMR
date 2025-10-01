
from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from genesis_viewer import GenesisViewer
import threading
import argparse

def main(args):
    genesis_env = GenesisViewer(visualize=True)

    from general_motion_retargeting.config.camera_config import Camera_Calibrations, World_Rotation
    genesis_env.initialize_cameras(Camera_Calibrations, World_Rotation)

    if args.get_offset:
        genesis_env.Real2Sim_offset_setup(args)
    else:
        genesis_env.Real2Sim_setup(args)

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

        if args.get_offset:
            offset = genesis_env.Real2Sim_offset_step(frame)
            if tic % 100 == 0:
                print(offset)
        else:
            genesis_env.Real2Sim_step(frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="192.168.0.232")
    parser.add_argument("--client_ip", type=str, default="192.168.0.128")
    parser.add_argument("--use_multicast", type=bool, default=False)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--get_offset", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
    