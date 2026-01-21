from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
import onnx

# 配置参数 (DistilBERT-base 默认配置)
# 如果你修改过模型结构，需要调整这两个参数
NUM_HEADS = 12
HIDDEN_SIZE = 768
MODEL_TYPE = 'distilbert' 

input_model = "model.onnx"
output_model = "model_fp16.onnx"

print(f"正在使用 Transformers 专用优化器优化 {input_model}...")

# 1. 运行优化器 (算子融合 + 图优化)
# 这一步会把零散的算子合并，大幅提升速度
optimized_model = optimizer.optimize_model(
    input_model,
    model_type=MODEL_TYPE,
    num_heads=NUM_HEADS,
    hidden_size=HIDDEN_SIZE,
    use_gpu=True
)

# 2. 转换为 FP16
# 专用优化器处理 FP16 时会自动插入正确的 Cast 节点，避免 Type Error
print("正在转换为 FP16...")
optimized_model.convert_float_to_float16()

# 3. 保存
optimized_model.save_model_to_file(output_model)
print(f"✅ 成功生成优化后的模型: {output_model}")
