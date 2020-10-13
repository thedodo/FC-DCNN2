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
  
## New trained weights

[Middlebury](https://drive.google.com/file/d/17LGGTQ0trAQs3qA05ITNXIHRoO65p-2n/view?usp=sharing) |
[Kitti2012](https://drive.google.com/file/d/19QLgLTDKtpqfuoAqUJBt4BhYPJdosyJ8/view?usp=sharing) |
[Kitti2015](https://drive.google.com/file/d/1mHZqw_xp3bXU2JOzgh6eFR-hznJunUxK/view?usp=sharing) |
[ETH3D](https://drive.google.com/file/d/1cnafA5Fupncdx9I_Yr1YfVDDL-SGTVmK/view?usp=sharing) |
[Sintel](https://drive.google.com/file/d/1Hg-DZGlnVkvbB-o9w4rrHIznx5UYd9Fn/view?usp=sharing) |
[drivingstereo](https://drive.google.com/file/d/1fXM6_dEkBL0qLNfpIYNr8nEsDDpYGKV6/view?usp=sharing)
