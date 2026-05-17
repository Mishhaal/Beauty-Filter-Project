import matplotlib.pyplot as plt

# -----------------------------
# Graph 1: Processing Time vs Number of Images
# -----------------------------
num_images = [50, 100, 150, 200]
total_time = [4.7, 9.4, 14.1, 18.7]

plt.figure()
plt.plot(num_images, total_time, marker='o')
plt.xlabel("Number of Images")
plt.ylabel("Total Processing Time (seconds)")
plt.title("Processing Time vs Number of Images")
plt.grid(True)

plt.savefig("processing_time_vs_images.png")  # ✅ SAVE IMAGE
plt.close()

# -----------------------------
# Graph 2: Average Processing Time per Image
# -----------------------------
avg_time = [0.094, 0.094, 0.094, 0.094]

plt.figure()
plt.bar(
    ["50 Images", "100 Images", "150 Images", "200 Images"],
    avg_time
)
plt.xlabel("Batch Size")
plt.ylabel("Average Time per Image (seconds)")
plt.title("Average Processing Time per Image")
plt.grid(True)

plt.savefig("avg_processing_time.png")  # ✅ SAVE IMAGE
plt.close()

print("✅ Graphs saved successfully:")
print(" - processing_time_vs_images.png")
print(" - avg_processing_time.png")
