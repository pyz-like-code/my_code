# 补充必要的导入
import onnxruntime as ort
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname=None, family="Microsoft YaHei", size=12)

# ===================== 1. NPU 环境配置（关键！优先调用锐龙NPU） =====================

# 自定义NPU执行器配置（适配锐龙NPU）
def get_npu_providers():
    """获取优先使用NPU的执行器列表"""
    providers = []
    # 1. 优先配置QNN执行器（锐龙NPU专属）
    try:
        qnn_options = {
            "device_id": 0,                # NPU设备ID（默认0）
            "backend_path": "",            # 留空自动识别NPU驱动
            "cache_dir": "./onnx_cache",   # NPU推理缓存目录
            "enable_caching": True         # 启用缓存提升推理速度
        }
        providers.append(("QNNExecutionProvider", qnn_options))
        print("✅ 已配置QNN执行器（锐龙NPU）")
    except Exception as e:
        print(f"⚠️ QNN执行器配置失败：{e}")
    
    # 2. 核显兜底（DirectML）
    try:
        providers.append("DmlExecutionProvider")
        print("✅ 已配置DirectML执行器（锐龙核显）")
    except:
        print("⚠️ DirectML执行器不可用")
    
    # 3. CPU兜底
    providers.append("CPUExecutionProvider")
    print("✅ 已配置CPU执行器（兜底）")
    
    return providers

# ===================== 2. 模型结构定义（和训练时保持一致） =====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
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

# ===================== 3. 初始化变量 =====================
device = torch.device("cpu")
model = SimpleCNN().to(device)

# 加载训练好的模型权重
try:
    model.load_state_dict(torch.load("mnist_cnn_amd.pth"))
    print("✅ 成功加载模型权重")
except FileNotFoundError:
    raise FileNotFoundError("❌ 未找到mnist_cnn_amd.pth，请先运行训练代码生成模型！")
except Exception as e:
    raise Exception(f"❌ 加载模型失败：{e}")

# 加载MNIST测试集（可视化用）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 初始化NPU推理会话（全局仅初始化一次，提升速度）
try:
    npu_providers = get_npu_providers()
    npu_sess = ort.InferenceSession("mnist_npu.onnx", providers=npu_providers)
    print(f"✅ NPU推理会话初始化成功，使用执行器：{npu_sess.get_providers()}")
except FileNotFoundError:
    # 自动转换PyTorch模型为ONNX（如果没有ONNX文件）
    print("⚠️ 未找到mnist_npu.onnx，自动从PyTorch模型转换...")
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model, dummy_input, "mnist_npu.onnx",
        input_names=["input"], output_names=["output"],
        opset_version=11, verbose=False
    )
    npu_sess = ort.InferenceSession("mnist_npu.onnx", providers=npu_providers)
    print("✅ ONNX模型转换完成，NPU推理会话初始化成功")
except Exception as e:
    raise Exception(f"❌ NPU推理会话初始化失败：{e}")

# ===================== 4. 纯NPU推理函数（核心修改） =====================
def predict_with_npu(image_path):
    """
    强制使用锐龙NPU推理手写数字（优先NPU，兜底核显/CPU）
    :param image_path: 手写数字图片路径
    :return: 识别结果（0-9）+ 使用的执行器
    """
    # 1. 图片预处理（严格匹配训练时的格式）
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ 无法读取图片：{image_path}")
    
    # 预处理步骤（和训练完全对齐）
    img = cv2.resize(img, (28, 28))          # 缩放到28x28
    img = 255 - img                          # 反转颜色（MNIST黑底白字）
    img = img / 255.0                        # 归一化到0-1
    img = (img - 0.1307) / 0.3081            # 标准化（替代torchvision的Normalize）
    img = np.expand_dims(img, axis=0)        # 增加通道维度 [28,28] → [1,28,28]
    img = np.expand_dims(img, axis=0)        # 增加批次维度 [1,28,28] → [1,1,28,28]
    img = img.astype(np.float32)             # NPU要求float32类型

    # 2. 纯NPU推理（使用全局初始化的NPU会话）
    try:
        output = npu_sess.run(["output"], {"input": img})[0]
        predicted = np.argmax(output)
        used_provider = npu_sess.get_providers()[0]  # 获取实际使用的执行器
        return int(predicted), used_provider
    except Exception as e:
        raise Exception(f"❌ NPU推理失败：{e}")

# ===================== 5. 可视化函数（兼容NPU推理结果） =====================
def visualize_prediction():
    """可视化MNIST测试集的NPU推理结果"""
    model.eval()
    try:
        data, target = next(iter(test_loader))
        # 批量转换为NPU推理格式
        batch_imgs = data.cpu().numpy().astype(np.float32)
        
        # NPU批量推理
        output = npu_sess.run(["output"], {"input": batch_imgs})[0]
        predicted = np.argmax(output, axis=1)
        
        # 绘制前5张
        fig, axes = plt.subplots(1, 10, figsize=(10, 3))
        for i in range(10):
            ax = axes[i]
            img_np = data[i].cpu().squeeze().numpy()
            ax.imshow(img_np, cmap='gray')
            ax.set_title(f'NPU预测: {predicted[i]}\n实际: {target[i].item()}', fontproperties=font)
            ax.axis('off')
        
        plt.suptitle(f"锐龙NPU推理结果（执行器：{npu_sess.get_providers()[0]}）", fontsize=12, fontproperties=font)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"❌ 可视化失败：{e}")

# ===================== 6. 测试运行 =====================
if __name__ == "__main__":
    # 1. 可视化NPU推理结果
    visualize_prediction()
    
    # 2. 测试单张图片的NPU推理（替换为你的手写数字图片路径）
    image_path = "handwritten_5.png"  # 请替换为实际路径
    try:
        result, used_provider = predict_with_npu(image_path)
        print(f"\n📌 手写数字识别结果：{result}")
        print(f"🔧 使用的执行器：{used_provider}")
        if "QNNExecutionProvider" in used_provider:
            print("✅ 成功使用锐龙NPU推理！")
        elif "DmlExecutionProvider" in used_provider:
            print("ℹ️ 使用锐龙核显推理（NPU未启用）")
        else:
            print("⚠️ 使用CPU推理（NPU/核显未识别）")
    except Exception as e:
        print(f"\n❌ 推理失败：{e}")