# Sign Language Recognition
<img width="380" height="133" alt="image" src="https://github.com/user-attachments/assets/f1260207-b02a-4173-bd78-7cdf6af07c7a" />

This repository contains the source code and resources for a Sign Language Recognition System. The aim of this project is to develop a computer vision system that can recognize and interpret sign language gestures in real-time, specifically American Sign Language (ASL). Each gesture is stored as a `.csv` file inside the `gesture_data` folder.

# Requirements / Installation
In order for this model to function, make sure you have **Python 3.7**+ installed and set up a virtual environment.
```
python -m venv myenv
myenv/Scripts/activate
```
Then, install the required library:
```
pip install torch
```
```
pip install opencv-python
```
```
pip install mediapipe
```
***Note:** You may go to https://pytorch.org/get-started/locally/ to install the correct version of PyTorch for your system if you are using a GPU and want to speed up the model training.*



# Reference
* [MediaPipe](https://mediapipe.dev/)
