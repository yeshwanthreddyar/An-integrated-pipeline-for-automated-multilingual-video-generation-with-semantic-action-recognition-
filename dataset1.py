# dataset1.py
import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UCF101Dataset(Dataset):
    def __init__(self, root_dir, classes_file, num_frames=16, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.samples = []

        # Load class names safely (handles both "1 ApplyEyeMakeup" and "ApplyEyeMakeup")
        with open(classes_file, 'r') as f:
            classes = []
            for line in f:
                parts = line.strip().split(' ', 1)
                classes.append(parts[-1])

        # ✅ ADD THIS LINE
        self.classes = classes  # required by dataset3.py

        # Map class names to numeric indices
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        # Scan dataset folders
        for label in classes:
            class_dir = os.path.join(root_dir, label)
            if not os.path.isdir(class_dir):
                print(f"⚠️ Skipping missing folder: {class_dir}")
                continue
            for file in os.listdir(class_dir):
                if file.endswith('.avi') or file.endswith('.mp4'):
                    path = os.path.join(class_dir, file)
                    self.samples.append((path, self.class_to_idx[label]))

        print(f"✅ Found {len(self.samples)} videos across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self.load_video_frames(video_path)
        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])
        else:
            frames = torch.stack(frames)
        return frames, label

    def load_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // self.num_frames)
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
            if len(frames) == self.num_frames:
                break
        cap.release()
        if len(frames) == 0:
            raise ValueError(f"No frames found in video: {path}")
        while len(frames) < self.num_frames:
            frames.append(frames[-1])  # pad if short video
        return frames
