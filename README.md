# AirType ✋✍️  
**Air Writing, Gesture Control & Vision-Based Interaction System**

AirType is a **computer vision–based input system** that enables **air writing**, **hand gesture recognition**, and **gesture-driven interaction** using a standard webcam — no physical input devices required.

The project is built using **MediaPipe Hands (Solutions API)** and **OpenCV**, focusing on real-time performance and modular design.

---

## 🚀 Features

- Real-time **hand tracking** using 21 landmark points  
- **Air writing** through finger trajectory tracking  
- **Gesture recognition** for interaction and control  
- Virtual canvas rendering with image export  
- Modular, multi-phase architecture for easy extension  

---

## 🛠 Tech Stack

- **Python 3.10 (mandatory)**
- **MediaPipe Hands (Solutions API)**
- **OpenCV**
- **NumPy**
- TensorFlow Lite (used internally by MediaPipe only)

---

## ⚠️ Important Note (Read Before Installing)

❌ **Do NOT install full TensorFlow in this environment**

- MediaPipe internally uses **TensorFlow Lite**
- Installing full **TensorFlow** causes **protobuf dependency conflicts**
- This project **does NOT require TensorFlow training APIs**

✔️ Use **MediaPipe Solutions only**

---

## ✅ System Requirements

- OS: Windows 10 / 11  
- Python: **3.10.x only**  
- Webcam: Any standard webcam  

---

## ⚡ Quick Start — Commands Only

> Run the following commands **in order** from the project root (`AirType/`).

- **Check installed Python versions**
  ```powershell
  py -0
  # 1. Clone the repository
  git clone https://github.com/ashh106/AirType.git
  cd AirType
  # 2. Create a virtual environment
  python -m venv .venv
  # 3. Activate the virtual environment
  # Windows (PowerShell)
  .venv\Scripts\Activate.ps1

  # Windows (CMD)
  .venv\Scripts\activate

  # macOS / Linux
  source .venv/bin/activate
  # 4. Upgrade pip
  python -m pip install --upgrade pip
  # 5. Install dependencies
  pip install -r requirements.txt
  # 6. Run the project (example: phase 1)
  python phase1_hand_tracking.py
  # 7. Deactivate the virtual environment (when done)
  deactivate
