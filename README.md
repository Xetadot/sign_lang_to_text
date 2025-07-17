# Sign Language Recognition
<img width="1000" height="150" alt="image" src="https://github.com/user-attachments/assets/f1260207-b02a-4173-bd78-7cdf6af07c7a" />

This repository contains the source code and resources for a Sign Language Recognition System. The aim of this project is to develop a computer vision system that can recognize and interpret sign language gestures in real-time, specifically American Sign Language (ASL). This is a beginner friendly guide on how to do so.

# Visual Studio Code (VS Code) Installation and Setup
We will be using `Visual Studio Code` as our main platform to run this model.
 1. Install [Visual Studio Code](https://code.visualstudio.com/) to your local device.
    
 2. Download the latest version of [Python](https://www.python.org/downloads/) and run the installer. **MAKE SURE** you check the box that says `Add Python to PATH` before clicking `Install Now`.
 3. Launch `Visual Studio Code`, click on `Extensions` on the left panel, search for `Python` and install the extension.

 4. On the bottom right, click on `Select Intepreter` and select the `Python` version that you've installed earlier.

 Hence, your `VS Code` setup is ready.

 # Code Setup
 Now that your `VS Code` is ready to be used, it's time to setup our code. We will need to create a virtual environment and install a few packages beforehand.
  1. In `VS Code`, click on `File` > `New File...` on the top left and name it as `model.py`. This is where the main code will be stored and runned.
  2. Open the terminal`(Ctrl + ~)` and run:
     ```
     python -m venv myenv
     ```
     This will create a folder named `myenv` with your virtual environment. `myenv` can be replaced with any other name you wish.

  3. Activate the environment by running:
     ```
     .\myenv\Scripts\activate
     ```
     After this, your terminal should show `(myenv)` in green - this means you are in the virtual environment.

  4. Install the required libraries:
     ```
     pip install torch
     ```
     ```
     pip install numpy
     ```
     ```
     pip install opencv-python
     ```
     ```
     pip install mediapipe
     ```
     ***Note:** You may go to https://pytorch.org/get-started/locally/ to install the correct version of PyTorch for your system if you are using a GPU (e.g. NVIDIA) and want to speed up the model training.*
     
  5. Create a folder called `gesture_data` in the same directory as the Python file. Inside it, add one `.csv` file per gesture. Each file should contain only rows of **63 float values** that represent 3D coordinates from `MediaPipe`.
     
     Example structure:
     ```
     gesture_classifier.py
     gesture_data/
     ├── Hello.csv
     ├── Thank you.csv
     ├── Goodbye.csv
     ```
 6. In the terminal, run:
    ```
    python gesture_classifier.py
    ```




# Reference
* [MediaPipe](https://mediapipe.dev/)
