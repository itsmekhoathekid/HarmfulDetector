from models.models import CustomModel, ResNet3DModel, DenseNet3D, R2Plus1DModel, ResNet, BasicBlock, VideoClassifier
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

# def collate_fn(batch):
#     # Loại bỏ các mục None khỏi batch
#     batch = [item for item in batch if item is not None]
#     # Ghép batch thành các tensor
#     video_frames, texts, spec_frames, labels = zip(*batch)
    
#     # Chuyển đổi texts (sen_embed) thành tensor
#     texts = torch.stack(texts) if isinstance(texts[0], torch.Tensor) else torch.tensor(texts, dtype=torch.float32)
    
#     return torch.stack(video_frames), texts, torch.stack(spec_frames), torch.tensor(labels)

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
        shuffle=True
    )

def get_inplanes():
    return [64, 128, 256, 512]

train_data = load_dataset("train")
test_data = load_dataset("test")
val_data = load_dataset("val")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CustomModel(sequence_length=n_frames, num_classes=5).to(device)
# model = ResNet3DModel(num_classes=5).to(device)
# model = DenseNet3D(
#     input_shape=(3, n_frames, 224, 224),  # [C, D, H, W]
#     num_blocks=3,
#     num_layers_per_block=4,
#     growth_rate=16,
#     dropout_rate=0.4,
#     compress_factor=0.5,
#     num_classes=5  # Số nhãn
# ).to(device)
model = R2Plus1DModel(num_classes=5).to(device)
# model = ResNet(
#     block=BasicBlock,  # Khối cơ bản của ResNet
#     layers=[2, 2, 2, 2],  # ResNet-18
#     block_inplanes=get_inplanes()  # Số nhãn (thay đổi theo dữ liệu của bạn)
# ).to(device)
model = VideoClassifier(num_classes=5).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(10):
    model.train()
    progress_bar = tqdm(train_data, desc=f"Epoch {epoch+1} Training", leave=True)
    for batch in progress_bar:
        video_frames, sen_embed, spec_frames, labels = batch
        print(video_frames)
        print(sen_embed)
        print(spec_frames)
        print(labels)
        video_frames, sen_embed, spec_frames, labels = video_frames.to(device), sen_embed.to(device), spec_frames.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(video_frames, sen_embed)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        progress_bar.set_description(f"Epoch {epoch+1} Training - Loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for batch in tqdm(val_data, desc="evaluating"):
            video_frames, texts, spec_frames, labels = batch
            video_frames, spec_frames, labels = video_frames.to(device), spec_frames.to(device), labels.to(device)
            outputs = model(video_frames)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Validation Accuracy: {correct / total}")
    
    


