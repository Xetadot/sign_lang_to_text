# AI Legal Assistant for the Rakyat
<img width="800" height="200" alt="asl" src="https://cdn.prod.website-files.com/672b8fcccce3fc53bb92fb97/672bab1defe652cd000583cc_1.png" />

This repository contains the source code and resources for an AI Legal Assistant. The aim of this project is to develop an AI-powered legal assistant that can:

* Summarize legal documents in plain English / Bahasa Melayu.

* Highlight key clauses (e.g., termination, payment, liabilities).

* Flag potential risks with easy-to-understand warnings.

* Provide context by grounding answers in Malaysian laws (via Retrieval-Augmented Generation).

# Visual Studio Code (VS Code) Installation and Setup
We will be using `Visual Studio Code` as our main platform to run this model.
 1. Install [Visual Studio Code](https://code.visualstudio.com/) to your local device.
    
 2. Download [Python](https://www.python.org/downloads/) and run the installer. **MAKE SURE** you check the box that says `Add Python to PATH` before clicking `Install Now`.
    
    _**Note:** Python version 3.11 is most ideal_
    
 4. Launch `Visual Studio Code`, click on `Extensions` on the left panel, search for `Python` and install the extension.

 5. On the bottom right, click on `Select Intepreter` and select the `Python` version that you've installed earlier.

 Hence, your `VS Code` setup is ready.

 # Code Setup
 Now that your `VS Code` is ready to be used, it's time to setup our code. We will need to create a virtual environment and install a few packages beforehand.

  1. Download the ZIP FILE
<img width="619" height="387" alt="codezip" src="https://bpb-us-e1.wpmucdn.com/sites.northwestern.edu/dist/b/3044/files/2021/05/github.png" />

  2. Extract "sign_lang_to_text" from zip file to download.

  3. Open the Extracted folder.

  4. Open the terminal`(Ctrl + ~)` and run:
     ```
     python -m venv myenv
     ```
     This will create a folder named `myenv` with your virtual environment. `myenv` can be replaced with any other name you wish.

  5. Activate the environment by running:
     ```
     \myenv\Scripts\activate
     ```
     After this, your terminal should show `(myenv)` in green - this means you are in the virtual environment.

  6. Install the required libraries:
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
     
# Running the Model
  Now that our code setup is done, we can finally run the model.

  1. Run `live_recognition.py` and allow `Python` to access your device's camera. A seperate window will appear.
  2. Show a gesture towards the camera. The model will translate the gesture showing `gesture:`.

     <img width="481" height="384" alt="image" src="https://github.com/user-attachments/assets/b2d5b893-3aa4-414e-b66a-820e55137b82" />

     _The gesture above translates into `Hello`._

  3. Press `Q` to exit the window and end the model.

     

     



     

     

     
 
    
      




# Reference
* [MediaPipe](https://mediapipe.dev/)
