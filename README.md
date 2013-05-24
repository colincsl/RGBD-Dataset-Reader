pyVideoDatasets
===============

Datasets using depth+color+skeleton information can often be cumbersome to work with. Everybody seems to use a different convention for each dataset. In this library each player allows you to to seemlessly stream depth/color/skeletal information. There is a single interface for accessing all of this data while hiding the IO from the user. 

This code can also be useful for looking at how to read in other datasets. For example, most of the MSR datasets have similar data formats -- I only have a player for MSRDailyActivity3D but it should be easy to modify the code to work with their others.

Feel free to email me at colincsl@gmail.com with any questions.

Install
-----------
sudo python setup.py install

Streaming capability (Uses BasePlayer)
-----------

MHADPlayer: Berkely Multimodal Human Activity Dataset

tele-immersion.citris-uc.org/berkeley_mhad/


MSRDailyActivity3D

http://research.microsoft.com/en-us/um/people/zliu/actionrecorsrc/


EVALPlayer: Stanford Skeleton Tracking Dataset (ECCV 2012)

http://ai.stanford.edu/~varung/eccv12/


pyKinectToolsPlayer: For use with my kinect library. Loads saved data.

https://github.com/colincsl/pyKinectTools


RealtimePlayer: For use with my kinect library. Streams data from a Kinect in realtime (requires pyOpenNI)

https://github.com/colincsl/pyKinectTools


Limited functionality/documentation
-----------

These are useful for seeing how to load the data, but aren't fully functional. 

ChalearnBodyPartReader: Chalearn gesture recognition competition data
http://gesture.chalearn.org/data
This file extracts depth images for all annotated joint data in the chalearn datasets.

CAD60Player: The Cornell Human Activities Dataset (CAD-60)
http://pr.cs.cornell.edu/humanactivities/data.php

SMMCPlayer: Stanford Skeleton Tracking Dataset (CVPR 2010)
http://ai.stanford.edu/~varung/cvpr10/
