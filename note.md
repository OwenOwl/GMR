### MoCap

1. optitrack_vendor/NatNetClient.py, __create_data_socket, since our lab is using broadcast on the server side, we have to receive everything from port 1511.

2. Remember to change offset in NatNetClient if see unidentified rigids.

3. Only camera pos needs xyzw to wxyz and y-up to z-up. rb poses are given in z-up and rolled in natnet

### AMASS

### Genesis

1. Currently using 29 dofs xml from GMR, removed xml scene

2. Genesis rebuild the kinematic tree, it is neccessary to reorder data pose or specify dofs_idx_local.

3. Set a large Max_FPS or the viewer will lock the entire FPS.