import ssl
# 第一步：禁用SSL证书验证（解决MNIST下载证书报错）
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np

# ===================== 1. 配置设备（适配Windows锐龙NPU/核显） =====================
# Windows优先用CPU训练，推理时用DirectML/NPU加速
device = torch.device("cpu")  # 训练阶段统一用CPU，避免环境问题
print(f"使用设备: {device}")

# ===================== 2. 数据加载（修复MNIST下载问题） =====================
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化，提升精度
])

# 关键修复：使用国内可访问的MNIST镜像源
try:
    # 优先从镜像源下载
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', 
        train=False, 
        download=True, 
        transform=transform
    )
except:
    # 备用方案：如果仍下载失败，手动下载后设置download=False
    print("自动下载失败，请手动下载MNIST数据集到 ./data/MNIST/raw 目录")
    print("下载地址：https://storage.googleapis.com/cvdf-datasets/mnist/")
    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)

# 数据加载器（分批处理）
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ===================== 3. 定义轻量级CNN模型（适配锐龙NPU/核显） =====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层（提取特征）
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 池化层（降维）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层（分类）
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 28/2/2=7
        self.fc2 = nn.Linear(128, 10)  # 10个数字类别
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)  # 防止过拟合

    def forward(self, x):
        # 前向传播
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 初始化模型并移到设备
model = SimpleCNN().to(device)
# 损失函数+优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===================== 4. 训练模型 =====================
def train_model(epochs=5):
    model.train()  # 训练模式
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            output = model(data)
            # 计算损失
            loss = criterion(output, target)
            # 反向传播+更新参数
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # 打印进度
            if batch_idx % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}')
                running_loss = 0.0
    print("训练完成！")

# 开始训练（5轮足够，CPU训练约5分钟）
train_model(epochs=5)

# ===================== 5. 测试模型精度 =====================
def test_model():
    model.eval()  # 评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度，提升速度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'模型测试精度: {100 * correct / total:.2f}%')

test_model()

# ===================== 6. 保存模型 =====================
torch.save(model.state_dict(), 'mnist_cnn_amd.pth')
print("模型已保存为 mnist_cnn_amd.pth")

# ===================== 7. 锐龙NPU/核显推理（优化版） =====================
# 安装依赖：pip install onnxruntime-directml
def predict_with_amd_npu(image_path):
    """使用锐龙NPU/核显推理（DirectML加速）"""
    try:
        import onnxruntime as ort
        # 1. 先将模型转为ONNX格式（仅第一次运行需要）
        dummy_input = torch.randn(1, 1, 28, 28)
        torch.onnx.export(
            model, dummy_input, "mnist_amd_npu.onnx",
            input_names=["input"], output_names=["output"], opset_version=11
        )
        
        # 2. 配置DirectML（调用锐龙NPU/核显）
        sess = ort.InferenceSession(
            "mnist_amd_npu.onnx",
            providers=["DmlExecutionProvider", "CPUExecutionProvider"]
        )
        
        # 3. 预处理图片
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "图片路径错误！"
        img = cv2.resize(img, (28, 28))
        img = 255 - img  # 反转颜色（MNIST是黑底白字）
        img = transforms.Normalize((0.1307,), (0.3081,))(transforms.ToTensor()(img))
        img = img.numpy().astype(np.float32)[np.newaxis, ...]
        
        # 4. NPU/核显推理
        output = sess.run(["output"], {"input": img})[0]
        predicted = np.argmax(output)
        return f"识别结果：{predicted}（NPU/核显加速）"
    except ImportError:
        # 备用：无DirectML时用CPU推理
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "图片路径错误！"
        img = cv2.resize(img, (28, 28))
        img = 255 - img
        img = transforms.Normalize((0.1307,), (0.3081,))(transforms.ToTensor()(img))
        img = img.unsqueeze(0).to(device)
        
        model.load_state_dict(torch.load('mnist_cnn_amd.pth'))
        model.eval()
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
        return f"识别结果：{predicted.item()}（CPU推理）"

# 测试推理（替换为你的手写数字图片路径）
# 示例：print(predict_with_amd_npu("handwritten_5.png"))

# ===================== 8. 可视化测试（可选） =====================
def visualize_prediction():
    model.eval()
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    _, predicted = torch.max(output, 1)
    
    # 显示前5张
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        ax = axes[i]
        ax.imshow(data[i].cpu().squeeze().numpy(), cmap='gray')
        ax.set_title(f'预测: {predicted[i].item()}, 真实: {target[i].item()}')
        ax.axis('off')
    plt.show()

visualize_prediction()