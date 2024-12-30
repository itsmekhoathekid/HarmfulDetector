import random
import numpy as np
import torch
from torchvision import transforms
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json

def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
    """
    # Convert frame to a tensor and normalize to [0, 1]
    frame = torch.from_numpy(frame).float() / 255.0
    frame = transforms.ToPILImage()(frame.permute(2, 0, 1))
    
    # Resize and pad the frame
    transform = transforms.Compose([
        transforms.Resize(output_size, antialias=True),
        transforms.Pad((0, 0, max(0, output_size[1] - frame.size[1]), max(0, output_size[0] - frame.size[0]))),
        transforms.ToTensor()
    ])
    return transform(frame)

def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
        frame: Image that needs to resized and padded.
        output_size: Tuple specifying (height, width) of the output frame image.

    Returns:
        Formatted frame with padding of specified output size.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
    frame = frame / 255.0  # Normalize to [0, 1]
    return frame

class JSONDataset(Dataset):
    def __init__(self, json_path, n_frames=16, frame_step=15, output_size=(224, 224), label_map=None):
        """
        Dataset class for loading data from JSON metadata.
        
        Args:
            json_path: Path to the JSON file containing dataset information.
            n_frames: Number of frames to extract from each video.
            frame_step: Step size between frames.
            output_size: Output frame size (height, width).
            label_map: Optional mapping from string labels to integers.
        """
        with open(json_path, "r", encoding="utf-8") as f:  # Specify utf-8 encoding
            self.data = json.load(f)
        self.n_frames = n_frames
        self.frame_step = frame_step
        self.output_size = output_size
        self.label_map = label_map if label_map else self._generate_label_map()

    def _generate_label_map(self):
        labels = set(item["label"] for item in self.data.values())
        return {label: idx for idx, label in enumerate(sorted(labels))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[str(idx)]
        video_path = Path(r'C:\Users\VIET HOANG - VTS\Desktop\VisionReader\HarmfulDetector') / entry.get("video_path")
        spec_path = Path(r'C:\Users\VIET HOANG - VTS\Desktop\VisionReader\HarmfulDetector') / entry.get("spec_path")
        text_embed = entry.get("embed_text", [])
        text_embed = torch.tensor(text_embed, dtype=torch.float32)
        label = self.label_map[entry["label"]]

        video_frames = self._extract_frames(video_path)
        mfcc_feats = entry.get("mfcc", [])
        mfcc_feats = torch.tensor(mfcc_feats, dtype=torch.float32)

        # Chuẩn hóa khung hình
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        video_frames = torch.stack([
            normalize(torch.tensor(frame).permute(2, 0, 1)) for frame in video_frames
        ])
        
        return video_frames, text_embed, mfcc_feats, torch.tensor(label)

    def _extract_frames(self, file_path):
        """Extract frames from a video or spectrogram file."""
        cap = cv2.VideoCapture(str(file_path))
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        need_length = 1 + (self.n_frames - 1) * self.frame_step
        start_frame = max(0, random.randint(0, frame_count - need_length) if need_length <= frame_count else 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(self.n_frames):
            ret, frame = cap.read()
            if ret:
                formatted_frame = format_frames(frame, self.output_size)
                # Chuẩn hóa khung hình
                normalized_frame = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])(
                    torch.tensor(formatted_frame).permute(2, 0, 1)
                )
                frames.append(normalized_frame.numpy().transpose(1, 2, 0))
                for _ in range(self.frame_step):
                    cap.read()  # Skip frames
            else:
                frames.append(np.zeros((*self.output_size, 3), dtype=np.float32))  # Black frame as placeholder

        cap.release()
        return np.array(frames, dtype=np.float32)

# class JSONDataset:
#     def __init__(self, json_path, n_frames=16, frame_step=15, output_size=(224, 224), label_map=None):
#         with open(json_path, "r", encoding="utf-8") as f:  # Specify utf-8 encoding
#             self.data = json.load(f)
#         self.label_map = label_map
#         self.n_frames = n_frames
#         self.frame_step = frame_step
#         self.output_size = output_size
#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         self.label_map = label_map if label_map else self._generate_label_map()

#     def _generate_label_map(self):
#         labels = set(item["label"] for item in self.data.values())
#         return {label: idx for idx, label in enumerate(sorted(labels))}

#     def __getitem__(self, idx):
#         entry = self.data[str(idx)]
#         video_path = Path(r'C:\Users\VIET HOANG - VTS\Desktop\VisionReader\HarmfulDetector') / entry.get("video_path")
#         spec_path = Path(r'C:\Users\VIET HOANG - VTS\Desktop\VisionReader\HarmfulDetector') / entry.get("spec_path")
#         text = entry.get("text", "")
#         label = self.label_map[entry["label"]]

#         # Chuẩn hóa và trích xuất khung hình
#         video_frames = self._extract_frames(video_path)
#         spec_frames = self._extract_frames(spec_path)

#         return video_frames, text, spec_frames, torch.tensor(label, dtype=torch.long)
    
#     def __len__(self):
#         return len(self.data)
    
#     def _extract_frames(self, file_path):
#         """Extract and normalize frames from a video or spectrogram file."""
#         cap = cv2.VideoCapture(str(file_path))
#         frames = []
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         # Xác định khung hình bắt đầu
#         need_length = 1 + (self.n_frames - 1) * self.frame_step
#         start_frame = max(0, random.randint(0, frame_count - need_length) if need_length <= frame_count else 0)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

#         for _ in range(self.n_frames):
#             ret, frame = cap.read()
#             if ret:
#                 formatted_frame = self.format_frame(frame)
#                 frames.append(formatted_frame)
#                 # Bỏ qua một số khung hình giữa các bước
#                 for _ in range(self.frame_step):
#                     cap.read()
#             else:
#                 # Thêm khung hình đen làm placeholder
#                 frames.append(torch.zeros((3, *self.output_size), dtype=torch.float32))

#         cap.release()
#         return torch.stack(frames)

#     def format_frame(self, frame):
#         """Format and normalize a single frame."""
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
#         frame = cv2.resize(frame, self.output_size, interpolation=cv2.INTER_AREA)  # Resize
#         frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Scale to [0, 1]
#         frame = self.normalize(frame)  # Chuẩn hóa
#         return frame