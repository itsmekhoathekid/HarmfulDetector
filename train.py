from models.models import CustomModel, ResNet3DModel, DenseNet3D, R2Plus1DModel, ResNet, BasicBlock, VideoTextClassifier
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
model = VideoTextClassifier(150,150,100,5,5,atten_type).to(device) 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
best_accuracy = 0.0

i = 0
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
for epoch in range(10):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_data, desc=f"Epoch {epoch+1} Training", leave=True)
    for batch in progress_bar:
        video_frames, sen_embed, mfcc_feats, labels = batch
        video_frames, sen_embed, mfcc_feats, labels = video_frames.to(device), sen_embed.to(device), mfcc_feats.to(device), labels.to(device)
        
        if atten_type == "multihead":
            labels = labels.squeeze(0)
        # labels = labels.squeeze(0)

        optimizer.zero_grad()
        outputs = model(video_frames, sen_embed, mfcc_feats)
        # print(labels)
        # print(outputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradient
        optimizer.step()

        progress_bar.set_description(f"Epoch {epoch+1} Training - Loss: {loss.item():.4f}")
    
    scheduler.step()

    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for batch in tqdm(val_data, desc="Evaluating"):
            video_frames, sen_embed, mfcc_feats, labels = batch

            # Chuyển dữ liệu sang thiết bị
            video_frames = video_frames.to(device)
            sen_embed = sen_embed.to(device)
            mfcc_feats = mfcc_feats.to(device)
            labels = labels.to(device)  # [batch_size]

            # Dự đoán
            outputs = model(video_frames, sen_embed, mfcc_feats)  # [batch_size, num_classes]
            labels.squeeze(0)

            # Lấy dự đoán với xác suất cao nhất
            _, predicted = torch.max(outputs, dim=0)  # [batch_size]

            # Cập nhật số lượng đúng và tổng
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Tính toán Accuracy
        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), r"C:\Users\VIET HOANG - VTS\Desktop\VisionReader\HarmfulDetector\checkpoint\best_model.pth")
        print("Model improved, saving model.")
