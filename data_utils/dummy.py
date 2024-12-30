import torch
import torch.nn as nn
# from model.model import TriModalModel
from utils import JSONDataset
from torch.utils.data import DataLoader
# Parameters
n_frames = 16  # Number of frames per video
output_size = (224, 224)  # Frame dimensions
frame_step = 15  # Frames to skip between samples
batch_size = 8
json_path = r"C:\Users\VIET HOANG - VTS\Desktop\VisionReader\HarmfulDetector\train.json"
dataset = JSONDataset(json_path=json_path, n_frames=n_frames, frame_step=frame_step, output_size=output_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


for batch in dataloader:
    video_frames, texts, spec_frames, labels = batch
    print("Video Frames Shape:", video_frames.shape)
    print("Text Samples:", texts)
    print("Spectrogram Frames Shape:", spec_frames.shape)
    print("Labels Shape:", labels.shape)
    print("Labels:", labels)
    break