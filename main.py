import os
import json
import random
from collections import defaultdict
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModel,  # 改为基础模型
    TrainingArguments,
    Trainer,
    PreTrainedTokenizer
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


class TripletDataset:
    """三元组数据集类"""

    def __init__(self, processed_data):
        self.data = processed_data
        self._build_label_index()

    def _build_label_index(self):
        """按标签分类存储文本索引"""
        self.label_to_indices = defaultdict(list)
        for idx, item in enumerate(self.data):
            self.label_to_indices[item["label"]].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """返回一个三元组（anchor, positive, negative）"""
        anchor_item = self.data[index]
        anchor_label = anchor_item["label"]

        # 随机选择同标签样本作为正样本
        positive_indices = [i for i in self.label_to_indices[anchor_label] if i != index]
        positive_idx = random.choice(positive_indices) if positive_indices else index
        positive_item = self.data[positive_idx]

        # 随机选择不同标签样本作为负样本
        negative_label = 1 - anchor_label
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_item = self.data[negative_idx]

        return {
            "anchor_text": anchor_item["text"],
            "positive_text": positive_item["text"],
            "negative_text": negative_item["text"],
            "anchor_label": anchor_label
        }


class TripletDataCollator:
    """三元组数据整理器"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """处理批次数据"""
        anchors = [item["anchor_text"] for item in batch]
        positives = [item["positive_text"] for item in batch]
        negatives = [item["negative_text"] for item in batch]

        # 使用tokenizer批量处理
        anchor_batch = self.tokenizer(
            anchors, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        positive_batch = self.tokenizer(
            positives, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        negative_batch = self.tokenizer(
            negatives, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        return {
            "anchor_input_ids": anchor_batch["input_ids"],
            "anchor_attention_mask": anchor_batch["attention_mask"],
            "positive_input_ids": positive_batch["input_ids"],
            "positive_attention_mask": positive_batch["attention_mask"],
            "negative_input_ids": negative_batch["input_ids"],
            "negative_attention_mask": negative_batch["attention_mask"],
            "labels": torch.tensor([item["anchor_label"] for item in batch])
        }


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
        """加载和预处理数据"""
        # 加载原始数据
        train_data = DataProcessor.load_data(train_path)
        test_data = DataProcessor.load_data(test_path)

        # 预处理数据
        processed_train = DataProcessor.preprocess_data(train_data)
        processed_test = DataProcessor.preprocess_data(test_data)

        print(f"训练集样本数: {len(processed_train)}")
        print(f"测试集样本数: {len(processed_test)}")

        return processed_train, processed_test

    def create_triplet_datasets(self, processed_train, processed_test):
        """创建三元组数据集"""
        triplet_train = TripletDataset(processed_train)
        triplet_test = TripletDataset(processed_test)

        # 转换为HF Dataset格式
        def convert_to_hf_dataset(triplet_dataset):
            data_dict = {
                "anchor_text": [],
                "positive_text": [],
                "negative_text": [],
                "anchor_label": []
            }

            for i in range(len(triplet_dataset)):
                item = triplet_dataset[i]
                data_dict["anchor_text"].append(item["anchor_text"])
                data_dict["positive_text"].append(item["positive_text"])
                data_dict["negative_text"].append(item["negative_text"])
                data_dict["anchor_label"].append(item["anchor_label"])

            return Dataset.from_dict(data_dict)

        hf_train = convert_to_hf_dataset(triplet_train)
        hf_test = convert_to_hf_dataset(triplet_test)

        return hf_train, hf_test

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
        collator = TripletDataCollator(self.tokenizer)

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
        """计算评估指标"""
        accuracy = evaluate.load("accuracy")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

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
        )

        self.trainer = HardTripletTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=TripletDataCollator(self.tokenizer),
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
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 初始化训练器
    trainer = ModelTrainer("distilbert-base-uncased")
    print("---初始化训练器已完成---")

    # 加载和预处理数据
    processed_train, processed_test = trainer.load_and_preprocess_data(
        'data/train.json',
        'data/test.json'
    )
    print("---加载与预处理数据已完成---")

    # 创建三元组数据集
    train_dataset, test_dataset = trainer.create_triplet_datasets(
        processed_train, processed_test
    )
    print("---Triplets 三元组数据集已完成---")

    # 设置模型和分词器
    trainer.setup_model_and_tokenizer()
    # 将模型移动到GPU
    if torch.cuda.is_available():
        trainer.model = trainer.model.cuda()
        print("---模型已移动到GPU---")
    else:
        print("---使用CPU训练---")
    print("---设置模型和分词器已完成---")

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