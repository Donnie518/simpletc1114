import os
import json
import random
from collections import defaultdict
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModel,  # 改为基础模型
    TrainingArguments,
    Trainer
)
import evaluate
import numpy as np

# 设置环境变量
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class DataProcessor:
    """数据处理类"""

    @staticmethod
    def load_data(file_path):
        """加载JSON数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def preprocess_data(original_data):
        """数据预处理"""
        processed_data = []

        for item in original_data:
            model_name = item.get("model", "")
            query = item.get("query", "")
            is_right = item.get("is_right", False)

            if not model_name or not query or is_right is None:
                continue

            text = f"Model: {model_name} Query: {query}"
            label = 1 if is_right else 0

            processed_data.append({
                "text": text,
                "label": label
            })

        return processed_data


import torch
from torch.nn.utils.rnn import pad_sequence


# ----------------- 优化部分开始 -----------------

class OptimizedTripletDataset(torch.utils.data.Dataset):
    """
    优化后的数据集：
    1. 接收已经分词好的 input_ids (list of list)
    2. 提前构建好正负样本索引，避免getitem时的查找开销
    """

    def __init__(self, tokenized_data):
        self.data = tokenized_data
        # 预先构建标签索引，加速采样
        self.label_to_indices = defaultdict(list)
        for idx, item in enumerate(self.data):
            self.label_to_indices[item["label"]].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor_item = self.data[index]
        anchor_label = anchor_item["label"]

        # 正样本采样
        positive_indices = self.label_to_indices[anchor_label]
        # 如果只有一个样本，就选自己
        positive_idx = random.choice(positive_indices) if len(positive_indices) > 1 else index
        # 避免选到自己（除非只有一个）
        while positive_idx == index and len(positive_indices) > 1:
            positive_idx = random.choice(positive_indices)

        positive_item = self.data[positive_idx]

        # 负样本采样
        negative_label = 1 - anchor_label
        if negative_label in self.label_to_indices:
            negative_idx = random.choice(self.label_to_indices[negative_label])
            negative_item = self.data[negative_idx]
        else:
            # 极端情况处理：如果没有负样本，随机选一个非自己的
            negative_item = self.data[random.randint(0, len(self.data) - 1)]

        # 直接返回 input_ids (列表形式)，不要在这里转 tensor，留给 collator 做
        return {
            "anchor_input_ids": anchor_item["input_ids"],
            "positive_input_ids": positive_item["input_ids"],
            "negative_input_ids": negative_item["input_ids"],
            "label": anchor_label
        }


class OptimizedTripletCollator:
    """
    优化后的整理器：
    实现动态 Padding，只 Pad 到当前 Batch 的最大长度，而不是 max_length
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # 提取数据
        anchor_ids = [torch.tensor(item["anchor_input_ids"]) for item in batch]
        pos_ids = [torch.tensor(item["positive_input_ids"]) for item in batch]
        neg_ids = [torch.tensor(item["negative_input_ids"]) for item in batch]
        labels = torch.tensor([item["label"] for item in batch])

        # 动态 Padding：使用 torch 自带的 pad_sequence
        # batch_first=True 会返回 [Batch, Seq_Len]
        anchor_batch = pad_sequence(anchor_ids, batch_first=True, padding_value=self.pad_token_id)
        pos_batch = pad_sequence(pos_ids, batch_first=True, padding_value=self.pad_token_id)
        neg_batch = pad_sequence(neg_ids, batch_first=True, padding_value=self.pad_token_id)

        # 生成 Attention Mask (非 pad 部分为 1)
        anchor_mask = (anchor_batch != self.pad_token_id).long()
        pos_mask = (pos_batch != self.pad_token_id).long()
        neg_mask = (neg_batch != self.pad_token_id).long()

        return {
            "anchor_input_ids": anchor_batch,
            "anchor_attention_mask": anchor_mask,
            "positive_input_ids": pos_batch,
            "positive_attention_mask": pos_mask,
            "negative_input_ids": neg_batch,
            "negative_attention_mask": neg_mask,
            "labels": labels
        }


# ----------------- 优化部分结束 -----------------

class TripletLoss(nn.Module):
    """自定义三元组损失"""

    def __init__(self, margin=1.0, distance_metric='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def pairwise_distance(self, x, y):
        """计算批次间的距离"""
        if self.distance_metric == 'euclidean':
            return torch.nn.functional.pairwise_distance(x, y)
        elif self.distance_metric == 'cosine':
            return 1 - torch.nn.functional.cosine_similarity(x, y)
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")

    def forward(self, anchor, positive, negative):
        if self.distance_metric == 'euclidean':
            pos_dist = torch.pow(anchor - positive, 2).sum(1)  # 欧氏距离平方
            neg_dist = torch.pow(anchor - negative, 2).sum(1)
            loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        else:  # cosine
            pos_dist = self.pairwise_distance(anchor, positive)
            neg_dist = self.pairwise_distance(anchor, negative)
            loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)

        return loss.mean()


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
    """结合三元组损失和分类损失的模型"""

    def __init__(self, model_name, num_labels=2, triplet_weight=0.3):
        super(TripletModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.classifier = ClassificationHead(self.hidden_size, num_labels)
        self.triplet_loss = TripletLoss(margin=0.5)
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_weight = triplet_weight

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                anchor_input_ids=None, anchor_attention_mask=None,
                positive_input_ids=None, positive_attention_mask=None,
                negative_input_ids=None, negative_attention_mask=None):

        # 如果有三元组输入，计算三元组损失
        if anchor_input_ids is not None:
            # 获取嵌入向量
            anchor_emb = self.get_sentence_embedding(anchor_input_ids, anchor_attention_mask)
            positive_emb = self.get_sentence_embedding(positive_input_ids, positive_attention_mask)
            negative_emb = self.get_sentence_embedding(negative_input_ids, negative_attention_mask)

            # 计算三元组损失
            triplet_loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)

            # 计算分类损失（使用anchor）
            anchor_logits = self.classifier(anchor_emb)
            ce_loss = self.ce_loss(anchor_logits, labels)

            # 组合损失
            total_loss = ce_loss + self.triplet_weight * triplet_loss

            return {
                'loss': total_loss,
                'logits': anchor_logits,
                'triplet_loss': triplet_loss,
                'ce_loss': ce_loss
            }
        else:
            # 普通分类模式
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            logits = self.classifier(embeddings)

            if labels is not None:
                loss = self.ce_loss(logits, labels)
                return {'loss': loss, 'logits': logits}
            else:
                return {'logits': logits}

    def get_sentence_embedding(self, input_ids, attention_mask):
        """获取句子嵌入"""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token


class HardTripletTrainer(Trainer):
    """自定义三元组训练器"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算组合损失"""
        # 调用模型的forward方法
        outputs = model(
            anchor_input_ids=inputs["anchor_input_ids"],
            anchor_attention_mask=inputs["anchor_attention_mask"],
            positive_input_ids=inputs["positive_input_ids"],
            positive_attention_mask=inputs["positive_attention_mask"],
            negative_input_ids=inputs["negative_input_ids"],
            negative_attention_mask=inputs["negative_attention_mask"],
            labels=inputs["labels"]
        )

        return (outputs['loss'], outputs) if return_outputs else outputs['loss']


class ModelTrainer:
    """模型训练主类"""

    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None

        # 标签映射
        self.id2label = {0: "INCORRECT", 1: "CORRECT"}
        self.label2id = {"INCORRECT": 0, "CORRECT": 1}

    def load_and_preprocess_data(self, train_path, test_path):
        """
        加载数据并【立即分词】
        """
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        train_data = DataProcessor.load_data(train_path)
        test_data = DataProcessor.load_data(test_path)

        # 定义一个简单的处理函数
        def process_and_tokenize(raw_data):
            processed = []
            # 设置较短的 max_length，例如 128
            # 如果你的文本真的很长，可以设为 256，但绝不要默认 512
            MAX_LEN = 128

            print("正在预分词数据...")
            for item in raw_data:
                model_name = item.get("model", "")
                query = item.get("query", "")
                is_right = item.get("is_right", False)
                if not model_name or not query or is_right is None:
                    continue

                text = f"Model: {model_name} Query: {query}"
                label = 1 if is_right else 0

                # 在这里直接分词，truncation=True 截断过长的
                # 注意：不要在这里 padding！我们留到 Collator 里做动态 padding
                tokenized = self.tokenizer(text, truncation=True, max_length=MAX_LEN)

                processed.append({
                    "input_ids": tokenized["input_ids"],  # 只要 input_ids
                    "label": label
                })
            return processed

        processed_train = process_and_tokenize(train_data)
        processed_test = process_and_tokenize(test_data)

        return processed_train, processed_test

    def create_triplet_datasets(self, processed_train, processed_test):
        """直接使用优化后的Dataset"""
        # 这里不需要转 HF Dataset 了，直接用我们自定义的 Torch Dataset 更灵活
        triplet_train = OptimizedTripletDataset(processed_train)
        triplet_test = OptimizedTripletDataset(processed_test)
        return triplet_train, triplet_test

    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 使用自定义的三元组模型
        self.model = TripletModel(
            model_name=self.model_name,
            num_labels=2,
            triplet_weight=0.3
        )

    def create_data_loaders(self, train_dataset, test_dataset, batch_size=16):
        """创建数据加载器"""
        collator = OptimizedTripletCollator(self.tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collator
        )

        return train_loader, test_loader

    def compute_metrics(self, eval_pred):
        """
        计算评估指标 (最终修复版)
        """
        predictions, labels = eval_pred

        # --- 核心修复代码 ---
        # 检查 predictions 是否为元组（因为模型返回了多个值）
        if isinstance(predictions, tuple):
            # 取第一个元素，也就是 logits
            predictions = predictions[0]
        # -------------------

        # 获取预测类别索引
        predictions = np.argmax(predictions, axis=1)

        # 使用 sklearn 计算准确率
        accuracy = accuracy_score(y_true=labels, y_pred=predictions)

        return {"accuracy": accuracy}

    def train(self, train_dataset, eval_dataset, output_dir="tri_model"):
        """训练模型"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            remove_unused_columns=False,  # 关键：不删除自定义列
            logging_steps=50,
            report_to=None,  # 禁用wandb等记录
            dataloader_num_workers=0,
        )

        self.trainer = HardTripletTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=OptimizedTripletCollator(self.tokenizer),
            compute_metrics=self.compute_metrics,
        )

        print("开始训练...")
        self.trainer.train()

    def evaluate(self, eval_dataset):
        """评估模型"""
        if self.trainer:
            return self.trainer.evaluate(eval_dataset)
        return None

def main():
    """主函数"""
    # 初始化训练器
    trainer = ModelTrainer("distilbert-base-uncased")

    # 加载和预处理数据
    processed_train, processed_test = trainer.load_and_preprocess_data(
        'data/train.json',
        'data/test.json'
    )

    # 创建三元组数据集
    train_dataset, test_dataset = trainer.create_triplet_datasets(
        processed_train, processed_test
    )

    # 设置模型和分词器
    trainer.setup_model_and_tokenizer()

    # 创建数据加载器（用于检查）
    train_loader, test_loader = trainer.create_data_loaders(train_dataset, test_dataset)

    # 检查一个batch
    sample_batch = next(iter(train_loader))
    print("Batch keys:", sample_batch.keys())
    print("Input IDs shape:", sample_batch["anchor_input_ids"].shape)

    # 训练模型
    trainer.train(train_dataset, test_dataset, "tri_model_1")


if __name__ == "__main__":
    main()
