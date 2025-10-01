from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from genesis_viewer import GenesisViewer
import threading
import argparse

def main(args):
    retarget = GMR(
            src_human="fbx",
            tgt_robot=args.robot,
            actual_human_height=1.85,
        )
    
    genesis_env = GenesisViewer()
    genesis_env.MoCap_setup(args=args, mujoco_model=retarget.model)

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

    print(f"OptiTrack client connected: {client.connected()}")
    print("Starting motion retargeting...")

    while True:
        frame = client.get_frame()
        frame_number = client.get_frame_number()
        qpos = retarget.retarget(frame)
        
        genesis_env.MoCap_step(qpos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="192.168.0.232")
    parser.add_argument("--client_ip", type=str, default="192.168.0.128")
    parser.add_argument("--use_multicast", type=bool, default=False)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    args = parser.parse_args()
    main(args)
    