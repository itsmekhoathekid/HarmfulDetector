from models.models import CustomModel, ResNet3DModel, DenseNet3D, R2Plus1DModel, ResNet, BasicBlock, BiLSTM, BiLSTM_classification
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

model = BiLSTM_classification(300, 150, 3, 5).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

i = 0
for epoch in range(10):
    model.train()
    progress_bar = tqdm(train_data, desc=f"Epoch {epoch+1} Training")
    for batch in progress_bar:
        video_frames, sen_embed, mfcc_feats, labels = batch
        
        # Định hình lại sen_embed
        sen_embed = sen_embed.unsqueeze(1).to(device)  # [batch_size, 1, 300]
        mfcc_feats = mfcc_feats.unsqueeze(1).to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(mfcc_feats)  # [batch_size, num_classes]
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix(loss=loss.item())
        
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Đánh giá
    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for batch in tqdm(val_data, desc="Validating"):
            _, sen_embed, mfcc_feats, labels = batch
            sen_embed = sen_embed.unsqueeze(1).to(device)  # [batch_size, 1, 300]
            mfcc_feats = mfcc_feats.unsqueeze(1).to(device)
            labels = labels.to(device)
            
            outputs = model(mfcc_feats)  # [batch_size, num_classes]
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Validation Accuracy: {correct / total:.4f}")
    