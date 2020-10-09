# FC-DCNN2
Updated version of the FC-DCNN network

The updates to the network are as follows: 

- color-patches are used instead of grayscale patches. This increases the complexity by about +1k parameters.
- TanH after output-layer removed
- during training:
  - patch-size has been increased from 11x11 to 21x21 
  - range for positive/negative patch has been increased from [2,6] to [1,25]
- new datasets:
  - KITTI has been trained seperately (2012, 2015)
  - Sintel has been added
  - drivingStereo has been added
