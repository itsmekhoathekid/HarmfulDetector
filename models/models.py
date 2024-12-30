import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models.video import r3d_18
from torchvision.models.video import r2plus1d_18
import torch.nn.functional as F
from functools import partial

class CustomModel(nn.Module):
    def __init__(self, sequence_length, num_classes, feature_extractor_freeze_layers=40):
        super(CustomModel, self).__init__()
        
        self.sequence_length = sequence_length  # Save sequence length
        
        # Load pretrained MobileNetV2
        mobilenet = mobilenet_v2(pretrained=True)
        
        # Remove the top classification layer
        for param in mobilenet.features[:-feature_extractor_freeze_layers].parameters():
            param.requires_grad = False
        self.feature_extractor = mobilenet.features
        
        # Specify TimeDistributed equivalent in PyTorch
        self.flatten = nn.Flatten()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=62720,  # Output of MobileNetV2 features
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(32 * 4, 256),  # Bidirectional LSTM output
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape # [1, 16, 3, 224, 224]
        
        # Reshape to combine batch and sequence dimensions
        x = x.view(batch_size * seq_len, c, h, w)  # Shape: [batch_size * seq_len, c, h, w] [16, 3, 224, 224]
        
        # Feature extraction with MobileNetV2
        features = self.feature_extractor(x)  # Shape: [batch_size * seq_len, 1280, h_out, w_out] 
        # print(features.shape)
        # Flatten spatial dimensions
        features = self.flatten(features)  # Shape: [batch_size * seq_len, 1280]
        # print(features.shape)
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, -1)  # Shape: [batch_size, seq_len, 1280]
        # print(features.shape)
        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # Shape: [batch_size, seq_len, hidden_size * 2]
        
        # Concatenate last forward and backward LSTM outputs
        lstm_out = torch.cat([lstm_out[:, -1, :32], lstm_out[:, 0, 32:]], dim=1)  # Shape: [batch_size, hidden_size * 2]
        
        # Fully connected layers
        out = self.fc_layers(lstm_out)  # Shape: [batch_size, num_classes]
        return out


class ResNet3DModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3DModel, self).__init__()
        
        # Load pretrained 3D ResNet (ResNet-18 for video)
        self.resnet3d = r3d_18(pretrained=True)  # Pretrained trên Kinetics-400
        
        # Thay đổi lớp Fully Connected cuối cùng
        in_features = self.resnet3d.fc.in_features
        self.resnet3d.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        # Đầu vào x có dạng [batch_size, C, D, H, W]
        # D: số khung hình, H x W: kích thước ảnh
        out = self.resnet3d(x)  # Đầu ra: [batch_size, num_classes]
        return out
    
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, dropout_rate, eps=1.1e-5):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate
        for i in range(num_layers):
            self.layers.append(self._dense_layer(in_channels + i * growth_rate, growth_rate, eps))

    def _dense_layer(self, in_channels, growth_rate, eps):
        return nn.Sequential(
            nn.BatchNorm3d(in_channels, eps=eps),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout3d(p=self.dropout_rate)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(x)
            features.append(out)
            x = torch.cat(features, dim=1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compress_factor, dropout_rate, eps=1.1e-5):
        super(TransitionLayer, self).__init__()
        num_output_channels = int(in_channels * compress_factor)
        self.transition = nn.Sequential(
            nn.BatchNorm3d(in_channels, eps=eps),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, num_output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Dropout3d(p=dropout_rate),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)


class DenseNet3D(nn.Module):
    def __init__(self, input_shape, num_blocks, num_layers_per_block, growth_rate, dropout_rate, compress_factor, num_classes, eps=1.1e-5):
        super(DenseNet3D, self).__init__()
        self.num_filters = 16
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate
        self.compress_factor = compress_factor

        self.initial_conv = nn.Conv3d(
            in_channels=input_shape[0],
            out_channels=self.num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(DenseBlock(num_layers_per_block, self.num_filters, self.growth_rate, self.dropout_rate, eps))
            self.num_filters += num_layers_per_block * self.growth_rate
            self.blocks.append(TransitionLayer(self.num_filters, self.compress_factor, self.dropout_rate, eps))
            self.num_filters = int(self.num_filters * self.compress_factor)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(self.num_filters, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class R2Plus1DModel(nn.Module):
    def __init__(self, num_classes):
        super(R2Plus1DModel, self).__init__()
        # Load pretrained R(2+1)D-18
        self.r2plus1d = r2plus1d_18(pretrained=True)  # Pretrained trên Kinetics-400
        # Thay thế lớp Fully Connected cuối cùng
        in_features = self.r2plus1d.fc.in_features
        self.r2plus1d.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        # Đầu vào x có dạng [batch_size, C, D, H, W]
        return self.r2plus1d(x)

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=5):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18

# Multi-Head Attention Class
class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)

    def forward(self, query, key, value):
        # Nếu đầu vào có batch, đưa về không batch
        if query.dim() == 3:  # [batch_size, seq_length, feature_dim]
            query = query.squeeze(0)  # [seq_length, feature_dim]
        if key.dim() == 3:
            key = key.squeeze(0)  # [seq_length, feature_dim]
        if value.dim() == 3:
            value = value.squeeze(0)  # [seq_length, feature_dim]

        # MultiheadAttention expects [seq_length, feature_dim]
        attended, _ = self.attention(query, key, value)
        return attended

# Modified R3D Model for Video Features
class ModifiedR3D(nn.Module):
    def __init__(self):
        super(ModifiedR3D, self).__init__()
        self.video_model = r3d_18(pretrained=True)
        self.video_model.fc = nn.Identity()  # Remove fully connected layer
        self.video_model.avgpool = nn.Identity()  # Remove global pooling

    def forward(self, x):
        x = self.video_model.stem(x)
        x = self.video_model.layer1(x)
        x = self.video_model.layer2(x)
        x = self.video_model.layer3(x)
        x = self.video_model.layer4(x)
        return x  # Feature map

# BiLSTM for Text Features
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)  # [batch_size, seq_length, hidden_size * 2]
        return lstm_out[:, -1, :]  # Take last hidden state

class ScaledDotProductAttention(nn.Module):
    def __init__(self, feature_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32))

    def forward(self, query, key, value, mask=None):

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # Áp dụng mask (nếu có)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Tính toán softmax
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Kết hợp attention weights với value
        attended_features = torch.matmul(attention_weights, value)
        return attended_features, attention_weights
    
# Main Model: Video-Text Classification
class VideoTextClassifier(nn.Module):
    def __init__(self, video_dim, text_dim, hidden_dim, num_heads, num_classes, attention_type):
        super(VideoTextClassifier, self).__init__()
        self.video_model = ModifiedR3D()
        self.text_model = BiLSTM(embedding_dim=300, hidden_size=hidden_dim, num_layers=2)
        self.mfcc_model = BiLSTM(embedding_dim=300, hidden_size=hidden_dim, num_layers=2)
        if attention_type == 'multihead':
            self.attention = MultiHeadAttention(feature_dim=hidden_dim * 2, num_heads=num_heads)
            self.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            self.attention = ScaledDotProductAttention(feature_dim=hidden_dim * 2)
            self.fc = nn.Sequential(
                nn.Linear(400, 200),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(200, num_classes)
            )
        # Linear layers to map to common space
        self.video_proj = nn.Linear(2, hidden_dim * 2)
        self.text_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.mfcc_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        # Final classifier
        
        self.atten_type = attention_type

    def forward(self, video_tensor, embed_sent, mfcc):
        # Video features
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # [batch_size, channels, frames, height, width]
        video_features = self.video_model(video_tensor)  # [batch_size, frames, feature_dim, h, w]
        video_features = video_features.mean(dim=[3, 4])  # [batch_size, frames, feature_dim]

        video_features = self.video_proj(video_features.mean(dim=0))  # [batch_size, hidden_dim * 2]

        # Text features
        embed_sent = embed_sent.unsqueeze(1)  # [batch_size, 1, embed_dim]
        text_features = self.text_model(embed_sent)  # [batch_size, hidden_dim * 2]
        text_features = self.text_proj(text_features)  # [batch_size, hidden_dim * 2]

        # MFCC features
        mfcc = mfcc.unsqueeze(1)
        mfcc_features = self.mfcc_model(mfcc)
        mfcc_features = self.mfcc_proj(mfcc_features)
        # Attention
        # print(video_features.unsqueeze(1).shape)
        # print(text_features.unsqueeze(1).shape)
        
        if self.atten_type == 'multihead':

            fused_features = self.attention(video_features, text_features.unsqueeze(1), text_features.unsqueeze(1))
            fused_features = fused_features.mean(dim=1, keepdim=False)
        else:
            attended_text, _ = self.attention(video_features, text_features.unsqueeze(1), text_features.unsqueeze(1))
            attended_mfcc, _ = self.attention(video_features, mfcc_features.unsqueeze(1), mfcc_features.unsqueeze(1))
            fused_features = torch.cat([attended_text, attended_mfcc], dim=-1)  # [batch_size, hidden_dim * 2]
            fused_features = fused_features.mean(dim=1) 
        
        # Classification
        logits = self.fc(fused_features)  # [batch_size, num_classes]
        return logits
    
class BiLSTM_classification(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers, num_classes):
        super(BiLSTM_classification, self).__init__()
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size*2, num_classes)
    def forward(self, x):
        lstm_out, _ = self.bilstm(x)  # [batch_size, seq_length, hidden_size * 2]
        logits = self.fc(lstm_out).mean(dim=1)
        return logits


class VideoTextClassifier2(nn.Module):
    def __init__(self, video_dim, text_dim, hidden_dim, num_heads, num_classes, attention_type):
        super(VideoTextClassifier2, self).__init__()
        self.video_model = ModifiedR3D()
        self.text_model = BiLSTM(embedding_dim=300, hidden_size=hidden_dim, num_layers=2)
        self.mfcc_model = BiLSTM(embedding_dim=300, hidden_size=hidden_dim, num_layers=2)
        if attention_type == 'multihead':
            self.attention = MultiHeadAttention(feature_dim=hidden_dim * 2, num_heads=num_heads)
            self.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            self.attention = ScaledDotProductAttention(feature_dim=hidden_dim * 2)
            self.fc = nn.Sequential(
                nn.Linear(200, 100),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(100, num_classes)
            )
        # Linear layers to map to common space
        self.video_proj = nn.Linear(2, hidden_dim * 2)
        self.text_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.mfcc_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        # Final classifier
        
        self.atten_type = attention_type

    def forward(self, video_tensor, embed_sent, mfcc):
        # Video features
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # [batch_size, channels, frames, height, width]
        video_features = self.video_model(video_tensor)  # [batch_size, frames, feature_dim, h, w]
        video_features = video_features.mean(dim=[3, 4])  # [batch_size, frames, feature_dim]

        video_features = self.video_proj(video_features.mean(dim=0))  # [batch_size, hidden_dim * 2]

        # Text features
        embed_sent = embed_sent.unsqueeze(1)  # [batch_size, 1, embed_dim]
        text_features = self.text_model(embed_sent)  # [batch_size, hidden_dim * 2]
        text_features = self.text_proj(text_features)  # [batch_size, hidden_dim * 2]

        # MFCC features
        mfcc = mfcc.unsqueeze(1)
        mfcc_features = self.mfcc_model(mfcc)
        mfcc_features = self.mfcc_proj(mfcc_features)
        # Attention
        # print(video_features.unsqueeze(1).shape)
        # print(text_features.unsqueeze(1).shape)
        
        if self.atten_type == 'multihead':

            fused_features = self.attention(video_features, text_features.unsqueeze(1), text_features.unsqueeze(1))
            fused_features = fused_features.mean(dim=1, keepdim=False)
        else:
            attended_text, _ = self.attention(video_features, text_features.unsqueeze(1), text_features.unsqueeze(1))
            attended_mfcc, _ = self.attention(video_features, mfcc_features.unsqueeze(1), mfcc_features.unsqueeze(1))
            attended_fused, _ = self.attention(attended_text, attended_mfcc, attended_mfcc)

            # fused_features = torch.cat([attended_text, attended_mfcc], dim=-1)  # [batch_size, hidden_dim * 2]
            attended_fused = attended_fused.mean(dim=1) 
        
        # Classification
        logits = self.fc(attended_fused)  # [batch_size, num_classes]
        return logits
    