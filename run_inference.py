import os
import torch
import time
from torch import nn
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

# ==================== 1. 模型类定义 (必须保持一致) ====================

class ClassificationHead(nn.Module):
    """分类头"""
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
    """推理用的模型类封装"""
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

# ==================== 2. 推理引擎 ====================

class InferenceEngine:
    def __init__(self, checkpoint_path, base_model_name="distilbert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")

        # 加载分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
           
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 初始化模型
        self.model = TripletModel(model_name=base_model_name)

        # 加载权重
        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
        bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"权重文件未找到: {checkpoint_path}")

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
       
        # 预热一次 (Warmup) 以避免第一次推理时间过长
        print("Warming up...")
        self.predict("test", "warmup")
        print("Model ready.")

    def predict(self, model_name, query):
        """执行推理并返回结果和耗时"""
        text = f"Model: {model_name} Query: {query}"
       
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        start_time = time.time()
       
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            logits = outputs['logits']
            prediction_idx = torch.argmax(logits, dim=-1).item()
           
            # 获取置信度
            probs = torch.softmax(logits, dim=-1)
            confidence = probs[0][prediction_idx].item()

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
       
        return {
            "result": True if prediction_idx == 1 else False,
            "latency": latency_ms,
            "confidence": confidence
        }

# ==================== 3. 主程序 ====================

def main():
    # --- 配置路径 ---
    # 请修改为你实际的 checkpoint 路径
    CHECKPOINT_PATH = "/root/autodl-tmp/tri_model_1/checkpoint-10590"
    BASE_MODEL = "/root/autodl-tmp/distilbert/distilbert-base-uncased"
    INPUT_FILE = "input.txt"
    OUTPUT_FILE = "output_result.txt"

    # 初始化
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到 {INPUT_FILE} 文件。请先创建该文件。")
        return

    try:
        engine = InferenceEngine(CHECKPOINT_PATH, BASE_MODEL)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    print(f"\n开始处理 {INPUT_FILE}...\n")
    print(f"{'Row':<4} | {'Result':<7} | {'Time(ms)':<8} | {'Conf':<6} | {'Content'}")
    print("-" * 80)

    results_buffer = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_time = 0
    count = 0

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
       
        # 解析 line: Model | Query
        # 使用 | 分割，允许 Query 中包含逗号
        parts = line.split('|', 1)
       
        if len(parts) < 2:
            print(f"{idx+1:<4} | ERROR   | 0.00     | 0.00   | 格式错误 (需要使用 | 分隔): {line}")
            continue

        model_name = parts[0].strip()
        query = parts[1].strip()

        # 推理
        res = engine.predict(model_name, query)
       
        # 打印到控制台
        res_str = "True" if res['result'] else "False"
        print(f"{idx+1:<4} | {res_str:<7} | {res['latency']:<8.2f} | {res['confidence']:.2f}   | {model_name} -> {query[:30]}...")

        # 保存结果
        output_line = f"Line {idx+1}: Result={res['result']}, Latency={res['latency']:.2f}ms, Model={model_name}, Query={query}\n"
        results_buffer.append(output_line)

        total_time += res['latency']
        count += 1

    # 写入结果文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(results_buffer)

    if count > 0:
        avg_time = total_time / count
        print("-" * 80)
        print(f"处理完成。共 {count} 条数据。")
        print(f"平均耗时: {avg_time:.2f} ms")
        print(f"详细结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
