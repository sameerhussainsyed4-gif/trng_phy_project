import cv2
import numpy as np
import time
import hashlib
from scipy.stats import chisquare
import matplotlib.pyplot as plt

x = int(input("How random do you want this (max frames): "))
n = int(input("How many random numbers do you want: "))

cap = cv2.VideoCapture(0)
print(cap.isOpened())
ret, prev_frames = cap.read()
if not ret:
    print("Frames are not found")
    cap.release()
    exit()

print("Please standby... capturing with frame-timing jitter...")

prev_frames = cv2.cvtColor(prev_frames, cv2.COLOR_BGR2GRAY)
height, width = prev_frames.shape

frames_count = 0
max_frames = x
all_bits = np.empty(max_frames * height * width, dtype=np.uint8)

entropy_pool = bytearray(time.time_ns().to_bytes(8, byteorder="big"))
generated_numbers = []

while frames_count < max_frames:

    t_start = time.perf_counter()
    ret, frames = cap.read()
    t_end = time.perf_counter()

    if not ret:
        break

    capture_duration_ns = int((t_end - t_start) * 1_000_000_000)

    start_idx = frames_count * height * width

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    motion_diff = cv2.absdiff(gray, prev_frames)
    lsb_matrix = motion_diff & 1

    all_bits[start_idx: start_idx + (height * width)] = lsb_matrix.ravel()

    entropy_pool.extend(time.time_ns().to_bytes(8, byteorder="big"))
    entropy_pool.extend(capture_duration_ns.to_bytes(
        8, byteorder="big", signed=False))
    entropy_pool.extend(
        all_bits[start_idx: start_idx + (height * width)].tobytes()[:128]
    )

    if len(entropy_pool) >= 128:
        digest = hashlib.sha256(entropy_pool).digest()
        uint32_vals = np.frombuffer(digest, dtype=np.uint32)
        random_numbers = uint32_vals % 1000
        generated_numbers.extend(random_numbers.tolist())

    prev_frames = gray
    frames_count += 1

cap.release()
print("Your Random Numbers Are:", generated_numbers[:n])


observed, _ = np.histogram(generated_numbers, bins=10, range=(0, 1000))
expected_freq = len(generated_numbers) / 10

chi2_stat, p_value = chisquare(observed, f_exp=expected_freq)

print(f"📊 Chi-Square Statistic: {chi2_stat:.4f}")
print(f"🧪 p-value: {p_value:.4f}")

plt.hist(generated_numbers, bins=10, range=(
    0, 1000), edgecolor='black', color='skyblue')
plt.title("Distribution of True Random Numbers (with Capture Timing Jitter)")
plt.xlabel("Value Range (0-1000)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
