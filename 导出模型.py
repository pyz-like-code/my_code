import torch
import torch.nn as nn

# 先定义和之前一样的模型结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 加载你之前训练好的权重
model = SimpleCNN()
model.load_state_dict(torch.load("mnist_cnn_amd.pth"))
model.eval()

# 导出为 ONNX
dummy_input = torch.randn(1, 1, 28, 28)  # 输入形状: [batch, channel, H, W]
torch.onnx.export(
    model,
    dummy_input,
    "mnist_npu.onnx",
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("模型已导出为 mnist_npu.onnx")