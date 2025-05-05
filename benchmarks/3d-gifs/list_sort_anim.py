import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === Step 1: Define swap matrix (genome) ===
list_length = 5
genome = np.random.randint(-1, 2, size=(list_length * list_length))
swap_matrix = genome.reshape(list_length, list_length)
swap_matrix = (swap_matrix > 0).astype(int)  # convert to binary: swap = 1

# === Step 2: Random test list ===
original_list = np.random.permutation(list_length)
test_list = original_list.copy()

# === Step 3: Bubble-sort style logic with swap control ===
steps = [test_list.copy()]

for i in range(list_length):
    for j in range(list_length - 1):
        if swap_matrix[test_list[j]][test_list[j + 1]] == 1:
            test_list[j], test_list[j + 1] = test_list[j + 1], test_list[j]
            steps.append(test_list.copy())  # store after every swap

# === Step 4: Plotting setup ===
fig, ax = plt.subplots()
bar = ax.bar(range(list_length), steps[0], color='skyblue')
ax.set_ylim(0, list_length)
title = ax.set_title("Step 0")
ax.set_xlabel("Index")
ax.set_ylabel("Value")

# === Step 5: Animation update function (fixed) ===
def update(i):
    frame = steps[i]
    for rect, val in zip(bar, frame):
        rect.set_height(val)
    title.set_text(f"Step {i}: {frame}")

# === Step 6: Animate and save ===
ani = FuncAnimation(fig, update, frames=len(steps), repeat=False)
ani.save("list_sort_animation.gif", dpi=120, writer="pillow")

print("GIF saved as list_sort_animation.gif")
