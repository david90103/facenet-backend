#### Dependencies
- Tensorflow 1.2.1 - gpu
- Python 3.5
- scipy
- scikit-learn
- opencv-python
- h5py
- matplotlib
- Pillow
- requests
- psutil

#### Setup
- Download pretrained model from drive, place in 20170511-185253/
- Download det1.py det2.py det3.py from davidsandberg repository, place in det/

#### Steps
1. put image "folders" in image/
2. run align.py
3. run classify.py
4. check VideoCapture path
5. run realtime.py


#### Questions
- Cannot match name with face in realtime detection
  - fixed: use re to filter class_name in model
- ValueError in align.py line 94
  - fixed: catch error and ignore that image
- Not accurate!!!!!
  - pretty accurate when reconizing classmates
- Only reconize one person when multi detection in one frame
  - fix by changing list index in realtime.py line 115 ~ 121
- 182 image size problem
