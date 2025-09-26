### MoCap

1. optitrack_vendor/NatNetClient.py, line 320, since our lab is using broadcast on the server side, we have to receive everything from port 1511.
2. Remember to change offset in NatNetClient if see unidentified rigids.

### AMASS

### Genesis

1. Currently using 29 dofs xml from GMR, removed xmll scene

2. Genesis rebuild the kinematic tree, it is neccessary to reorder data pose or specify dofs_idx_local.