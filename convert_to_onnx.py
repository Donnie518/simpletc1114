import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
import onnxruntime as ort
import numpy as np

# ==================== 1. 必须复制的模型定义 ====================
# (为了保证加载权重时结构一致，必须原样复制)

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class TripletModel(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(TripletModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.classifier = ClassificationHead(self.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(embeddings)
        return {'logits': logits}

# ==================== 2. ONNX 导出专用封装 ====================

class OnnxExportWrapper(nn.Module):
    """
    这一层封装是为了剥离字典输出，只返回 Tensor。
    方便 C++/Python 推理时直接获取 output[0]。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
   
    def forward(self, input_ids, attention_mask):
        # 调用原始模型
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # 只提取 logits (Tensor)
        return outputs['logits']

# ==================== 3. 转换主逻辑 ====================

def export_model():
    # --- 配置区域 ---
    CHECKPOINT_PATH = "/root/autodl-tmp/tri_model_1/checkpoint-10590" # 您的权重路径
    BASE_MODEL_NAME = "/root/autodl-tmp/distilbert/distilbert-base-uncased"                       # 您的基础模型
    ONNX_OUTPUT_PATH = "model.onnx"                                   # 输出文件名
   
    device = torch.device("cpu") # 导出时推荐使用 CPU，避免不必要的 CUDA 依赖
    print(f"1. 初始化模型 (Base: {BASE_MODEL_NAME})...")

    # A. 初始化模型
    model = TripletModel(model_name=BASE_MODEL_NAME)
   
    # B. 加载权重 (兼容 safetensors 和 bin)
    safetensors_path = os.path.join(CHECKPOINT_PATH, "model.safetensors")
    bin_path = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        print(f"   加载 Safetensors: {safetensors_path}")
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        print(f"   加载 PyTorch Bin: {bin_path}")
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"找不到权重文件在: {CHECKPOINT_PATH}")

    model.load_state_dict(state_dict, strict=True)
    model.eval() # 必须设为 eval 模式 (关闭 Dropout)

    # C. 包装模型 (去掉字典输出)
    wrapped_model = OnnxExportWrapper(model)

    # D. 准备 Dummy Input (伪造输入)
    # ONNX 需要运行一次模型来追踪计算图
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    dummy_text = "This is a sample query for tracing."
    inputs = tokenizer(dummy_text, return_tensors="pt")
   
    dummy_input_ids = inputs["input_ids"].to(device)
    dummy_attention_mask = inputs["attention_mask"].to(device)

    print("2. 开始导出 ONNX...")

    #

    # E. 执行导出
    torch.onnx.export(
        wrapped_model,                          # 要导出的模型
        (dummy_input_ids, dummy_attention_mask),# 输入元组
        ONNX_OUTPUT_PATH,                       # 保存路径
        export_params=True,                     # 导出权重矩阵
        opset_version=14,                       # 算子集版本 (建议 12-14)
        do_constant_folding=True,               # 优化：常量折叠
        input_names=['input_ids', 'attention_mask'], # 输入节点命名
        output_names=['logits'],                # 输出节点命名
        dynamic_axes={                          # 关键：支持动态 Batch 和 动态长度
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"   导出完成: {ONNX_OUTPUT_PATH}")

    # ==================== 4. 验证导出结果 ====================
    verify_onnx(wrapped_model, dummy_input_ids, dummy_attention_mask, ONNX_OUTPUT_PATH)

def verify_onnx(torch_model, dummy_ids, dummy_mask, onnx_path):
    print("3. 验证 ONNX 模型精度...")
   
    # PyTorch 结果
    with torch.no_grad():
        torch_out = torch_model(dummy_ids, dummy_mask).numpy()

    # ONNX Runtime 结果
    ort_session = ort.InferenceSession(onnx_path)
    onnx_inputs = {
        'input_ids': dummy_ids.numpy(),
        'attention_mask': dummy_mask.numpy()
    }
    onnx_out = ort_session.run(None, onnx_inputs)[0]

    # 对比
    if np.allclose(torch_out, onnx_out, rtol=1e-03, atol=1e-05):
        print(f"✅ 验证成功！结果一致。\n   PyTorch logits: {torch_out[0][:3]}\n   ONNX logits:    {onnx_out[0][:3]}")
    else:
        print("❌ 验证失败！结果差异较大。")

if __name__ == "__main__":
    export_model()
