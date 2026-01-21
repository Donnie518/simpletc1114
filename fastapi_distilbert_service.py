# fastapi_distilbert_service.py
# 一个显存 <200MB 的 DistilBERT 在线推理服务示例（RTX 4090 友好）

import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import uvicorn

# ==============================
# 1. CUDA / PyTorch 全局设置
# ==============================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEVICE = "cuda"
DTYPE = torch.float16
MODEL_NAME = "distilbert-base-uncased"

# ==============================
# 2. 加载 tokenizer（CPU 常驻）
# ==============================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ==============================
# 3. 加载模型（FP16 + FlashAttention）
# ==============================

model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    attn_implementation="flash_attention_2",
).to(DEVICE)

model.eval()

# ==============================
# 4. FastAPI 定义
# ==============================

app = FastAPI(title="DistilBERT Online Inference Service")


class InferenceRequest(BaseModel):
    text: str
    max_length: int = 256


class InferenceResponse(BaseModel):
    embedding: list
    dim: int


# ==============================
# 5. 推理函数（关键：inference_mode）
# ==============================

@torch.inference_mode()
def encode(text: str, max_length: int):
    # tokenizer 在 CPU
    inputs = tokenizer(
        text,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors="pt",
    )

    # 仅在这里把 tensor 放到 GPU
    input_ids = inputs["input_ids"].to(DEVICE, non_blocking=True)
    attention_mask = inputs["attention_mask"].to(DEVICE, non_blocking=True)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # 使用 [CLS] embedding（DistilBERT 用第一个 token）
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding.squeeze(0).float().cpu().tolist()


# ==============================
# 6. API 接口
# ==============================

@app.post("/encode", response_model=InferenceResponse)
def encode_api(req: InferenceRequest):
    embedding = encode(req.text, req.max_length)
    return InferenceResponse(
        embedding=embedding,
        dim=len(embedding),
    )


# ==============================
# 7. 启动服务
# ==============================

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_distilbert_service:app",
        host="0.0.0.0",
        port=8000,
        workers=1,   # ⚠️ 一个 GPU = 一个 worker
    )
