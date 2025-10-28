from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch as t
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import chess
import time
import mss
import pyautogui
import tkinter as tk
from pynput import keyboard
from pyfiglet import Figlet



BOARD_MODEL_PATH = "model_files/board_model_state.pth"
MOVE_MODEL_PATH  = "model_files/move_model_state.pth"
LABEL_PATH = "model_files/label.npy"

OVERLAY_W = 600
OVERLAY_H = 600
OVERLAY_X = 200
OVERLAY_Y = 150

LOOP_DELAY = 5     
CLICK_PAUSE = 0.06        
DRAG_INSTEAD_OF_CLICK = False  
COLOR_FLAG_FOR_MODEL = 1 



# %%
DEVICE = 'mps' if getattr(t.backends, "mps", None) and t.backends.mps.is_available() else ('cuda' if t.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# %%

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# %%

class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 13, 3, padding=1),
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = nn.functional.adaptive_avg_pool2d(x, (8,8))
        return x


# %%

class ChessImproved(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(14, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3), nn.ReLU(), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1792)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



# %%
def ten_con(fen):
    arr = np.zeros((13, 8, 8), dtype=np.float32)
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    board_part = fen.split()[0]
    ranks = board_part.split('/')
    for row, rank in enumerate(ranks):
        col = 0
        for ch in rank:
            if ch.isdigit():
                for _ in range(int(ch)):
                    arr[12, row, col] = 1
                    col += 1
            else:
                arr[piece_to_index[ch], row, col] = 1
                col += 1
    return arr



# %%
def tensor_to_fen_argmax(pred_2d):
    index_to_piece = {
        0:'P', 1:'N', 2:'B', 3:'R', 4:'Q', 5:'K',
        6:'p', 7:'n', 8:'b', 9:'r', 10:'q', 11:'k', 12: None
    }
    fen_rows = []
    for row in range(8):
        fen_row = ""
        empty_count = 0
        for col in range(8):
            square = int(pred_2d[row, col].item())
            if square == 12:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += index_to_piece[square]
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)

# %%
Label = np.load(LABEL_PATH, allow_pickle=True)
ans_sender = {}
inf_changer = {}
for cnt, i in enumerate(Label):
    ans_sender[cnt] = str(i)
    inf_changer[str(i)] = cnt

# %%
def vaild_checker(logits_tensor, fen):
    try:
        board_state = chess.Board(fen)
    except Exception:
        return False
    legal_set = set(m.uci() for m in board_state.legal_moves)
    logits = logits_tensor.clone().detach().cpu().squeeze()
    for idx, move_str in ans_sender.items():
        if move_str not in legal_set:
            logits[idx] = float('-inf')
    if (logits == float('-inf')).all():
        return False
    return logits


# %%
def uci_to_from_to(uci):
    u = uci.strip()
    return u[0:2], u[2:4]

# %%
def algebraic_to_pixel(square: str, top_left_x:int, top_left_y:int, board_width:int):
    file_chars = 'abcdefgh'
    file = square[0]
    rank = square[1]
    square_size = board_width / 8.0
    col = file_chars.index(file)
    row = 8 - int(rank)
    cx = int(top_left_x + col * square_size + square_size / 2)
    cy = int(top_left_y + row * square_size + square_size / 2)
    return cx, cy


# %%
board_model = ChessCNN().to(DEVICE)
move_model  = ChessImproved().to(DEVICE)

board_model.load_state_dict(t.load(BOARD_MODEL_PATH, map_location=DEVICE))
move_model.load_state_dict(t.load(MOVE_MODEL_PATH, map_location=DEVICE))
board_model.eval()
move_model.eval()

# %%
def predict_move_from_pil(pil_img: Image.Image, color_flag:int):
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)  
    with t.no_grad():
        y_pred = board_model(img_tensor)   
        _, pred = y_pred.max(dim=1)        
        pred_2d = pred[0].cpu()
        fen_board_str = tensor_to_fen_argmax(pred_2d)
        ten = ten_con(fen_board_str)
        extra = np.full((1,8,8), color_flag, dtype=np.float32)
        combined = np.concatenate([ten, extra], axis=0) 
        combined_t = t.from_numpy(combined).unsqueeze(0).to(DEVICE).float()
        logits = move_model(combined_t)  
        filtered = vaild_checker(logits, fen_board_str)
        if filtered is False:
            return None
        idx = int(t.argmax(filtered).item())
        uci = ans_sender[idx]
        return uci

# %%
capture_region = None

def create_resize_overlay():
    root = tk.Tk()
    root.title("🔲 Position & Resize Overlay → Press SPACE When Done")
    
    root.geometry("400x400+300+200")

    root.resizable(True, True)

    label = tk.Label(root, text="Resize and Move This Window\nThen Press SPACE", font=("Arial", 14), fg="white", bg="black")
    label.pack(fill="both", expand=True)

    def lock_position(event):
        x = root.winfo_x()
        y = root.winfo_y()
        w = root.winfo_width()
        h = root.winfo_height()

        print(f"Board Region Locked: ({x}, {y}, {w}, {h})")

        root.destroy()
        return (x, y, w, h)

    root.bind("<space>", lock_position)

    root.mainloop()


# # %%
# create_resize_overlay() 
""" Use this when you need to resize the screen size and run thaniya bcaz it cause thadangalgal !!!"""

# %%
def run_live_loop(region, color_flag=COLOR_FLAG_FOR_MODEL):
    left, top, width, height = region
    sct = mss.mss()
    last_move = None
    stop_flag = {"stop": False}

    def on_press(key):
        try:
            if key == keyboard.Key.esc:
                stop_flag["stop"] = True
                return False
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while not stop_flag["stop"]:
        shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
        
        frame = Image.fromarray(np.array(shot))
        frame = Image.fromarray(np.array(shot))

        frame.save("debug_last_board.png")

        
        uci = predict_move_from_pil(frame, color_flag)
        if uci is None:
            time.sleep(LOOP_DELAY)
            continue
        if uci == last_move:
            time.sleep(LOOP_DELAY)
            continue

        print("Predicted move:", uci)
        try:
            from_sq, to_sq = uci_to_from_to(uci)
            fx, fy = algebraic_to_pixel(from_sq, left, top, width)
            tx, ty = algebraic_to_pixel(to_sq, left, top, width)
            pyautogui.moveTo(fx, fy)
            pyautogui.click()
            time.sleep(CLICK_PAUSE)
            if DRAG_INSTEAD_OF_CLICK:
                pyautogui.mouseDown()
                pyautogui.moveTo(tx, ty, duration=0.12)
                pyautogui.mouseUp()
            else:
                pyautogui.moveTo(tx, ty)
                pyautogui.click()
            last_move = uci
        except Exception as e:
            print("Click/perform error:", e)

        time.sleep(LOOP_DELAY)

    print("Live loop stopped.")



# %%


# %%
print( "\033[91m" + Figlet(font="standard").renderText("Viyugam "+"64") + "\033[0m")
region = (32, 228, 692, 693)
if not region:
    print("No region selected. Exiting.")
    exit(0)
time.sleep(5)
run_live_loop(region, color_flag=COLOR_FLAG_FOR_MODEL)


