from sklearn.metrics import classification_report
from models.models import CustomModel, ResNet3DModel, DenseNet3D, R2Plus1DModel, ResNet, BasicBlock, VideoTextClassifierResNet
from data_utils.utils import JSONDataset
from torch.utils.data import DataLoader
import torch
import torch
from torch import nn
from tqdm import tqdm

n_frames = 10  # Number of frames per video
output_size = (224, 224)  # Frame dimensions
frame_step = 15  # Frames to skip between samples
batch_size = 1

json_paths = {
    "train" : r"C:\Users\VIET HOANG - VTS\Desktop\VisionReader\HarmfulDetector\train.json",
    "test" : r"C:\Users\VIET HOANG - VTS\Desktop\VisionReader\HarmfulDetector\test.json",
    "val" : r"C:\Users\VIET HOANG - VTS\Desktop\VisionReader\HarmfulDetector\val.json"
}

def collate_fn(batch):
    # Loại bỏ các mục None khỏi batch
    batch = [item for item in batch if item is not None]
    
    # Ghép batch thành các tensor
    video_frames, texts, spec_frames, labels = zip(*batch)
    
    # Chuyển đổi texts (sen_embed) thành tensor
    texts = torch.stack(texts) if isinstance(texts[0], torch.Tensor) else torch.tensor(texts, dtype=torch.float32)
    
    return torch.stack(video_frames), texts, torch.stack(spec_frames), torch.tensor(labels)

def load_dataset(typee):
    json_path = json_paths[typee]
    
    # Tạo dataset
    dataset = JSONDataset(
        json_path=json_path,
        n_frames=n_frames,
        frame_step=frame_step,
        output_size=output_size
    )
    
    # Tạo DataLoader
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

train_data = load_dataset("train")
test_data = load_dataset("test")
val_data = load_dataset("val")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

atten_type = "dotproduct" # 'multihead' , 'dotproduct'
checkpoint_path = r"C:\Users\VIET HOANG - VTS\Desktop\VisionReader\HarmfulDetector\checkpoint\best_model.pth"
model = VideoTextClassifierResNet(150, 150, 100, 5, 5, atten_type).to(device)
model.load_state_dict(torch.load(checkpoint_path))

model.eval()  # Chuyển mô hình sang chế độ đánh giá
with torch.no_grad():  # Không tính toán gradient trong chế độ đánh giá
    total_samples, correct_predictions = 0, 0
    all_labels = []
    all_predictions = []
    
    for batch in tqdm(test_data, desc="Evaluating"):
        video_frames, sen_embed, mfcc_feats, labels = batch

        # Chuyển dữ liệu sang thiết bị
        video_frames = video_frames.to(device)
        sen_embed = sen_embed.to(device)
        mfcc_feats = mfcc_feats.to(device)
        labels = labels.to(device)

        # Dự đoán
        outputs = model(video_frames, sen_embed, mfcc_feats)  # [batch_size, num_classes]

        # Lấy dự đoán với xác suất cao nhất
        _, predicted = torch.max(outputs, dim=1)  # [batch_size]
        # Cập nhật số lượng đúng và tổng
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Lưu kết quả để tính các metric chi tiết
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    # Tính toán Accuracy
    accuracy = correct_predictions / total_samples
    print(f"Testing Accuracy: {accuracy:.4f}")

# Báo cáo chi tiết các metric
from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=["pornographic", "normal", "offensive", "horrible", "violent"]))