# Sign Language Recognition
<img width="800" height="150" alt="image" src="https://github.com/user-attachments/assets/f1260207-b02a-4173-bd78-7cdf6af07c7a" />

This repository contains the source code and resources for a Sign Language Recognition System. The aim of this project is to develop a computer vision system that can recognize and interpret sign language gestures in real-time, specifically American Sign Language (ASL). This is a **beginner friendly** guide.

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
  1. In `VS Code`, click on `File` > `New File...` on the top left and name it as `model.py`, paste the following code:
     
      ```
     import os
     import csv
     import torch
     import torch.nn as nn
     from torch.utils.data import Dataset, DataLoader
     import numpy as np

     DATA_DIR = "gesture_data"
     GESTURE_LIST = [fname.split(".")[0] for fname in os.listdir(
         DATA_DIR) if fname.endswith(".csv")]
 

     class GestureDataset(Dataset):
         def __init__(self, data_dir, gesture_list):
             self.samples = []
             self.labels = []
             self.label_map = {gesture: idx for idx,
                               gesture in enumerate(gesture_list)}
             for gesture in gesture_list:
                 filepath = os.path.join(data_dir, f"{gesture}.csv")
                 with open(filepath, 'r') as f:
                     reader = csv.reader(f)
                     for row in reader:
                         if len(row) != 63:
                             continue
                         keypoints = np.array(row, dtype=np.float32)
                         self.samples.append(keypoints)
                         self.labels.append(self.label_map[gesture])

             self.samples = np.array(self.samples)
             self.labels = np.array(self.labels)

             print(
                 f"Loaded {len(self.samples)} samples for {len(gesture_list)} classes")

         def __len__(self):
             return len(self.samples)

         def __getitem__(self, idx):
             return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


     class GestureClassifier(nn.Module):
         def __init__(self, input_size=63, hidden_size=128, num_classes=len(GESTURE_LIST)):
             super(GestureClassifier, self).__init__()
             self.net = nn.Sequential(
                 nn.Linear(input_size, hidden_size),
                 nn.ReLU(),
                 nn.Dropout(0.3),
                 nn.Linear(hidden_size, hidden_size),
                 nn.ReLU(),
                 nn.Dropout(0.3),
                 nn.Linear(hidden_size, num_classes)
             )

         def forward(self, x):
             return self.net(x)

        
         def train():
             dataset = GestureDataset(DATA_DIR, GESTURE_LIST)
             dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

             model = GestureClassifier()
             criterion = nn.CrossEntropyLoss()
             optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
             epochs = 20
             for epoch in range(epochs):
                 model.train()
                 total_loss = 0
                 for data, labels in dataloader:
                     optimizer.zero_grad()
                     outputs = model(data)
                     loss = criterion(outputs, labels)
                     loss.backward()
                     optimizer.step()
                     total_loss += loss.item()

                 avg_loss = total_loss / len(dataloader)
                 print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

             torch.save(model.state_dict(), "gesture_classifier.pth")
             print("Model saved as gesture_classifier.pth")


      if __name__ == "__main__":
             train()
     ```
      This is where the main code will be stored.
     
  2. Open the terminal`(Ctrl + ~)` and run:
     ```
     python -m venv myenv
     ```
     This will create a folder named `myenv` with your virtual environment. `myenv` can be replaced with any other name you wish.

  3. Activate the environment by running:
     ```
     \myenv\Scripts\activate
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
     
  5. Create a folder `gesture_data` in the same directory as the Python file. Inside `gesture_data`, add one `.csv` file per gesture. Each file should contain only rows of **63 float values** that represent 3D coordinates from `MediaPipe`.
     
     Example structure:
     ```
     gesture_classifier.py
     gesture_data/
     ├── Hello.csv
     ├── Thank you.csv
     ├── Goodbye.csv
     ```
  6. Create a file `gesture_classifier.py` and paste the previous code from `model.py` and run it.
    
  7. Create a file `live_recognition.py` and paste the following code:
     ```
     import cv2
     import mediapipe as mp
     import torch
     import torch.nn as nn
     import numpy as np
     from datetime import datetime



     class GestureClassifier(nn.Module):
         def __init__(self, input_size=63, hidden_size=128, num_classes=5):
             super(GestureClassifier, self).__init__()
             self.net = nn.Sequential(
                 nn.Linear(input_size, hidden_size),
                 nn.ReLU(),
                 nn.Dropout(0.3),
                 nn.Linear(hidden_size, hidden_size),
                 nn.ReLU(),
                 nn.Dropout(0.3),
                 nn.Linear(hidden_size, num_classes)
             )

         def forward(self, x):
             return self.net(x)


     class_names = ["Hello", "Ok", "Thank you", "Goodbye", "Help"]

     model = GestureClassifier(input_size=63, hidden_size=128,
                               num_classes=len(class_names))
     model.load_state_dict(torch.load('gesture_classifier.pth',
                           map_location=torch.device('cpu')))
     model.eval()

     mp_hands = mp.solutions.hands
     hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
     mp_drawing = mp.solutions.drawing_utils

     cap = cv2.VideoCapture(0)
     spoken_gesture = None


     while True:
         ret, frame = cap.read()
         if not ret:
             break

         frame = cv2.flip(frame, 1)
         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         result = hands.process(rgb)

         if result.multi_hand_landmarks:
             for hand_landmarks in result.multi_hand_landmarks:
                 mp_drawing.draw_landmarks(
                     frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                 landmarks = []
                 for lm in hand_landmarks.landmark:
                     landmarks.extend([lm.x, lm.y, lm.z])

                 input_tensor = torch.tensor(
                     landmarks, dtype=torch.float32).unsqueeze(0)

                 with torch.no_grad():
                     output = model(input_tensor)
                     predicted_idx = torch.argmax(output, dim=1).item()
                     predicted_gesture = class_names[predicted_idx]
                     confidence = torch.softmax(output, dim=1)[
                         0][predicted_idx].item()

                     result_json = {
                         "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                         "gesture": predicted_gesture,
                         "confidence": round(confidence, 4)
                     }

                 cv2.putText(frame, f'Gesture: {predicted_gesture}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

         else:
             spoken_gesture = None

         cv2.imshow('Sign Language Recognition', frame)
         if cv2.waitKey(1) & 0xFF == ord('q'):
             break

     cap.release()
     cv2.destroyAllWindows()
     ```
  8. Create a file `manual_data.py` and paste the following code:
     ```
     import cv2
     import mediapipe as mp
     import csv
     import os

     gesture_list = ["Hello", "Ok", "Thank you", "Goodbye", "Help"]
     SAMPLES_PER_GESTURE = 1000
     DATA_DIR = "gesture_data"
     os.makedirs(DATA_DIR, exist_ok=True)

     mp_hands = mp.solutions.hands
     hands = mp_hands.Hands()
     draw = mp.solutions.drawing_utils

     cap = cv2.VideoCapture(0)

     for gesture in gesture_list:
         print(f"\n➡️ Show gesture: {gesture} – press Enter to begin")
         input()

         count = 0
         while count < SAMPLES_PER_GESTURE:
             success, frame = cap.read()
             frame = cv2.flip(frame, 1)
             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             result = hands.process(rgb)

             if result.multi_hand_landmarks:
                 for hand_landmarks in result.multi_hand_landmarks:
                     draw.draw_landmarks(frame, hand_landmarks,
                                         mp_hands.HAND_CONNECTIONS)

                     keypoints = []
                     for lm in hand_landmarks.landmark:
                         keypoints += [lm.x, lm.y, lm.z]

                     with open(f"{DATA_DIR}/{gesture}.csv", 'a', newline='') as f:
                         writer = csv.writer(f)
                         writer.writerow(keypoints)

                     count += 1

             cv2.putText(frame, f"{gesture}: {count}/{SAMPLES_PER_GESTURE}", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
             cv2.imshow("Collecting Gesture Data", frame)

             if cv2.waitKey(1) & 0xFF == ord('q'):
                 break

     print("\n✅ Data collection done.")
     cap.release()
     cv2.destroyAllWindows()
     ```
   
# Running the Model
  Now that our code setup is done, we can finally run the model.

  1. Run `live_recognition.py` and allow `Python` to access your device's camera. A seperate window will appear.
  2. Show a gesture towards the camera. The model will translate the gesture showing `gesture:`.

     <img width="481" height="384" alt="image" src="https://github.com/user-attachments/assets/b2d5b893-3aa4-414e-b66a-820e55137b82" />

     _The gesture above translates into `Hello`._

  3. Press `Q` to exit the window and end the model.

     

     



     

     

     
 
    
      




# Reference
* [MediaPipe](https://mediapipe.dev/)
