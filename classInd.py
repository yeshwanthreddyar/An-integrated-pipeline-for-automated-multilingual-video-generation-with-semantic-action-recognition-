import os

root = "C:/Users/Yeshwanth Reddy A R/Downloads/pibb/UCF-101"  # path to your dataset folder
classes = sorted(os.listdir(root))
with open(os.path.join(root, "classInd.txt"), "w") as f:
    for i, cls in enumerate(classes, 1):
        f.write(f"{i} {cls}\n")
print("✅ classInd.txt created successfully!")
