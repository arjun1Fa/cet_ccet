"""Quick diagnostic — run this before main.py to confirm everything works."""

import cv2, requests, time, os

print("=" * 55)
print("PIPELINE DIAGNOSTIC")
print("=" * 55)

# ── 1. Video capture test ──────────────────────────────────
print("\n[1/2] Video Capture Test...")
videos = [
    'data/stream_ (1).mp4',
    'data/stream_ (2).mp4',
    'data/stream_ (3).mp4',
    'data/stream_ (4).mp4',
]
all_ok = True
for path in videos:
    if not os.path.exists(path):
        print(f"  ✗ NOT FOUND: {path}")
        all_ok = False
        continue
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  ✗ Cannot open: {path}")
        all_ok = False
    else:
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ {path}  →  frame shape: {frame.shape}")
        else:
            print(f"  ✗ Opened but couldn't read frame: {path}")
            all_ok = False
    cap.release()

if all_ok:
    print("  → Video capture: OK ✅")
else:
    print("  → Video capture: FAILED ❌")

# ── 2. GPU server / SSH tunnel test ───────────────────────
print("\n[2/2] GPU Server / SSH Tunnel Test...")
url = "http://127.0.0.1:8000/health"
try:
    r = requests.get(url, timeout=5)
    data = r.json()
    print(f"  ✓ Server reachable!")
    print(f"    Status : {data.get('status')}")
    print(f"    GPU    : {data.get('gpu')}")
    print(f"    GPU Name: {data.get('gpu_name')}")
    print("  → GPU server: OK ✅")
except requests.exceptions.ConnectionError:
    print("  ✗ Cannot reach http://127.0.0.1:8000")
    print("  → Either SSH tunnel is not running OR server crashed ❌")
    print()
    print("  FIX:")
    print("  1. Open a new PowerShell and run the TUNNEL command:")
    print("     ssh -i id_ed25519_new -p 22130 -L 8000:localhost:8000 -N root@69.30.85.42")
    print("  2. Open another PowerShell, SSH into the pod and restart the server:")
    print("     ssh -i id_ed25519_new -p 22130 root@69.30.85.42")
    print("     then: pkill -9 python && python /workspace/gpu_server.py")
except requests.exceptions.Timeout:
    print("  ✗ Server timed out (tunnel up, but server not responding) ❌")
    print("  FIX: SSH into pod and restart gpu_server.py")

print("\n" + "=" * 55)
