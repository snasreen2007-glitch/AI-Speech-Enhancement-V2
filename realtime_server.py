import socket
import struct
import numpy as np
import torch
from model_V2 import DTLNNoiseCanceller

HOST = "0.0.0.0"
PORT = 5001
SR = 16000
N = 4000   # 500 ms chunks for initial reliable demo

model = DTLNNoiseCanceller()
ck = torch.load("checkpoints/FINAL_MODEL_V2.pt", map_location="cpu")
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval()

def recv_exact(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def enhance(x):
    with torch.no_grad():
        y = model(torch.from_numpy(x).float().view(1, 1, -1))
    if isinstance(y, (tuple, list)):
        y = y[0]
    return y.detach().cpu().numpy().reshape(-1).astype(np.float32)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)

    print(f"AI SERVER READY: {HOST}:{PORT}", flush=True)

    while True:
        conn, addr = s.accept()
        print("CONNECTED:", addr, flush=True)

        with conn:
            while True:
                header = recv_exact(conn, 4)
                if header is None:
                    break

                size = struct.unpack("<I", header)[0]
                raw = recv_exact(conn, size)

                if raw is None:
                    break

                x = np.frombuffer(raw, dtype=np.float32)

                try:
                    y = enhance(x)
                    conn.sendall(struct.pack("<I", len(y.tobytes())))
                    conn.sendall(y.tobytes())
                except Exception as e:
                    print("INFERENCE ERROR:", e, flush=True)
                    break

        print("CLIENT DISCONNECTED", flush=True)
