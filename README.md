# Drone_Paper



## ENV SETTING ##
	Windows 11
	Intel i7 11th, 12th
	Nvidia RTX3060 ,RTX3070, RTX3070ti
	python 3.8
	RAM 16GB, 32GB
	torch 1.13.1
	cuda 11.7
	cudnn 8.0
	
### git clone ###
```bash
git clone https://github.com/YouKyoung-Na/Drone_Paper.git
```

## Windows ENV Setting ##
### Anaconda ENV Setting ###
```bash
conda create -n 'Drone_Project' python=3.8
conda activate Dron_Project
cd Drone_Paper
```

### Install library(For use Nvidia GPU) ###
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
cd detector_tracker
pip install -r requirements.txt
pip install natsort
conda install -c conda-forge lap
pip install rembg[gpu]
cd ..
```


### weights 이름의 dir 생성 후 아래 pt file 집어넣기 ###


### pt file Download (detector_Tracker yolov5X size weight, 나중에 변경) ###
```python
https://www.notion.so/0406694cfafb4181be35fb49e8782abc#422455d8d2a44e9a9ea5a057df1ee363
```

## RUN ##
추후 합칠 예정(일단 각각 실행)

### Detector, Tracker(in detector_tracker dir) ###

#### Use Webcam ####
```bash
python SCIE.py --source 0 # cam number 0, 1, 2, ...
```

```bash
python SCIE.py --source (img_dir) # cam number 0, 1, 2, ...
```
