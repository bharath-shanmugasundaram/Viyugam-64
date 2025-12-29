<p align="center">
  <a href="#-model-architecture">
    <img src="https://img.shields.io/badge/Architecture-View-blue?style=for-the-badge" />
  </a>
  <a href="https://leetcode.com/u/bharath_shanmugasundaram/">
    <img src="https://img.shields.io/badge/How%20It%20Works-Explore-green?style=for-the-badge" />
  </a>
  <a href="#-future-enhancements">
    <img src="https://img.shields.io/badge/Future%20Plans-Roadmap-orange?style=for-the-badge" />
  </a>
</p>


# ♟️ Viyugam 64

**Viyugam 64** is an AI-powered Chess Automation system that **predicts and plays the next best move** directly on your computer’s chessboard screen using deep learning.  
It combines **Computer Vision + Deep Learning + Automation** to observe the board, predict the best move, and execute it using mouse control — just like a real player.

---
## Demo



https://github.com/user-attachments/assets/7a8472aa-64d1-4c0f-94e3-8da5d3889618


## 🚀 Features

- 🎯 **Board Recognition** – CNN-based model predicts the board’s current FEN state from a screenshot.  
- 🧠 **Move Prediction** – Deep network predicts the best legal chess move.  
- 🖱️ **Auto Move Execution** – Automatically moves the piece on screen using `pyautogui`.  
- 🪟 **Overlay Region Selection** – Simple `tkinter` window for board capture region setup.  
- 💻 **Multi-device Support** – Runs seamlessly on **MPS (Apple Silicon)**, **CUDA (NVIDIA GPU)**, or **CPU**.  
- 🔁 **Live Looping Mode** – Continuously predicts and plays until stopped (ESC key).  

---

## 🧩 Model Architecture

### 1️⃣ Board Recognition Model (`ChessCNN`)
Predicts the 8×8 board matrix from a screenshot image.

- Input: Grayscale 64×64 chessboard image  
- Output: 13×8×8 tensor (12 pieces + empty square channel)  
- Core Layers:
  - 4 convolutional layers  
  - ReLU activations  
  - Adaptive average pooling to (8×8)

### 2️⃣ Move Prediction Model (`ChessImproved`)
Takes the board tensor + color flag and predicts the best move.

- Input: (14×8×8) tensor  
- Output: 1792 move logits  
- Architecture:
  - Convolutional + BatchNorm blocks  
  - AdaptiveAvgPool2d  
  - Fully-connected layers → Dropout → Output  

---

## 🧠 Dataset Format

- **Board images**: Captured screenshots of live chess games.  
- **FEN labels**: Represent board states.  
- **Move labels**: Indexed UCI moves (`a2a4`, `b1c3`, etc.).  
- **`label.npy`**: Contains all possible move mappings.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/bharath-shanmugasundaram/Viyugam-64.git
cd Viyugam-64
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**Example requirements:**
```text
torch
torchvision
numpy
pillow
mss
pyautogui
python-chess
tk
pynput
pyfiglet
```

---

## 📦 Model Files

Make sure these are present inside the `model_files/` directory:

```
model_files/
 ├── board_model_state.pth
 ├── move_model_state.pth
 └── label.npy
```

---

## 🧭 How It Works

1. The program captures your chessboard screen region using `mss`.  
2. The **Board Model** predicts the board’s FEN structure.  
3. The **Move Model** predicts the next best move.  
4. It validates the move against legal moves using `python-chess`.  
5. The move is executed via `pyautogui` mouse control.

---

## ▶️ Usage

### 1️⃣ Set Board Region
Optionally, run this function to manually resize the capture window:
```python
create_resize_overlay()
```
Press **SPACE** once aligned.

### 2️⃣ Run Prediction
At the bottom of the script:
```python
region = (32, 228, 692, 693)
time.sleep(5)
run_live_loop(region, color_flag=1)
```

- `color_flag = 1` → You play **White**  
- `color_flag = 0` → You play **Black**  

Press **ESC** anytime to stop the loop.

---

## 🧾 Output Example

```
Using device: mps
Predicted move: e2e4
Predicted move: d7d5
Predicted move: g1f3
...
```

---

## 🖼️ Debugging

To debug the captured board image:
```bash
debug_last_board.png
```
This file saves the most recent captured board frame.

---

## 🧱 Project Structure

```
Viyugam-64/
├── model_files/
│   ├── board_model_state.pth
│   ├── move_model_state.pth
│   └── label.npy
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚡ Controls

| Key | Action |
|-----|---------|
| `ESC` | Stop live loop |
| `SPACE` | Lock overlay region |

---

## 💡 Future Enhancements

- ✅ Real-time multi-angle board detection  
- ✅ Reinforcement learning–based move improvement  
- ✅ GUI dashboard with move history  
- ✅ Online chess platform integration  

---

## 🧠 Inspiration

> “Viyugam” (வியூகம்) means **Strategy** in Tamil —  
> inspired by the strategic depth of chess and modern AI decision-making.

---

## 🧑‍💻 Author

**Bharath Shanmugasundaram**  
AI Engineer | Deep Learning Enthusiast  
📍 India  
🔗 [GitHub Profile](https://github.com/bharath-shanmugasundaram)

---

## 🏷️ License

This project is licensed under the **MIT License** – feel free to use, modify, and share.

---

## 🌟 Support

If you like this project, please ⭐ the repository — it really helps!
