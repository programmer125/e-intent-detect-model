# AGENTS.md

## 1. 项目概览（tree）

```text
e-intent-detect-model
├── 目标
│   ├── 电商二级意图识别（单意图输出）
│   ├── 线上兜底：OTHER
│   └── 性能目标：RTX 4090 上 P95 < 50ms
├── 技术路线
│   ├── 训练：Encoder 分类微调（当前 smoke 训练脚本）
│   ├── 推理：ONNX Runtime GPU
│   └── 服务：FastAPI（Python 直接启动）
└── 环境基线
    ├── Ubuntu 22.04
    ├── CUDA 11.8
    ├── Python 3.10
    └── PyTorch 2.4.1+cu118
```

## 2. 仓库结构（tree）

```text
/Users/duyx/workspace/JSAI/e-intent-detect-model
├── app.py                                # 在线推理服务
├── environment.yml                       # conda 基础环境
├── requirements-cu118.txt                # 依赖清单（CUDA 11.8）
├── intent
│   └── common.json                       # 意图映射与反向映射
├── data
│   ├── intent_train_1k.train.jsonl       # 训练集
│   ├── intent_train_1k.val.jsonl         # 验证集
│   ├── intent_train_1k.test.jsonl        # 测试集
│   ├── intent_hard_negatives.jsonl       # 难负样本
│   └── intent_confusion_set.jsonl        # 混淆样本
└── scripts
    ├── setup_conda_env.sh                # 一键建环境
    ├── verify_gpu.py                     # CUDA 可用性检查
    ├── smoke_train.py                    # 训练冒烟（默认 100 step）
    ├── benchmark_inference.py            # 延迟/QPS基准
    ├── export_onnx.py                    # 导出 ONNX
    └── check_onnx_consistency.py         # PyTorch vs ONNX 一致性
```

## 3. 环境与依赖（tree）

```text
环境要求
├── OS: Ubuntu 22.04
├── CUDA: 11.8
├── Driver: >= 520（建议 >= 535）
└── Conda Env: intent-llm

关键依赖
├── torch==2.4.1 (cu118)
├── transformers==4.44.2
├── onnx==1.16.2
├── onnxruntime-gpu==1.16.3   # 与 CUDA 11.8 兼容
└── modelscope==1.18.1        # 无法访问 HuggingFace 时使用
```

## 4. 标准执行流程（tree）

```text
执行流程
├── 1) 创建环境
│   ├── conda env create -f /root/gpufree-data/e-intent-detect-model/environment.yml
│   ├── conda activate intent-llm
│   └── pip install -r /root/gpufree-data/e-intent-detect-model/requirements-cu118.txt
├── 2) GPU 校验
│   └── python /root/gpufree-data/e-intent-detect-model/scripts/verify_gpu.py
├── 3) 训练冒烟
│   └── python /root/gpufree-data/e-intent-detect-model/scripts/smoke_train.py --train_path /root/gpufree-data/e-intent-detect-model/data/intent_train_1k.train.jsonl --val_path /root/gpufree-data/e-intent-detect-model/data/intent_train_1k.val.jsonl --test_path /root/gpufree-data/e-intent-detect-model/data/intent_train_1k.test.jsonl --num_train_epochs 12 --model_source modelscope --model_name damo/nlp_structbert_backbone_base_std --output_dir /root/gpufree-data/e-intent-detect-model/outputs/model_v2
├── 4) 推理性能测试
│   └── python /root/gpufree-data/e-intent-detect-model/scripts/benchmark_inference.py --model_dir /root/gpufree-data/e-intent-detect-model/outputs/model_v2 --input_path /root/gpufree-data/e-intent-detect-model/data/intent_train_1k.test.jsonl --num_samples 1000 --batch_size 1 --max_length 64
├── 5) 导出与一致性验证
│   ├── python /root/gpufree-data/e-intent-detect-model/scripts/export_onnx.py --model_dir /root/gpufree-data/e-intent-detect-model/outputs/model_v2 --onnx_path /root/gpufree-data/e-intent-detect-model/outputs/model_v2/model.onnx --max_length 64
│   └── python /root/gpufree-data/e-intent-detect-model/scripts/check_onnx_consistency.py --model_dir /root/gpufree-data/e-intent-detect-model/outputs/model_v2 --onnx_path /root/gpufree-data/e-intent-detect-model/outputs/model_v2/model.onnx --input_path /root/gpufree-data/e-intent-detect-model/data/intent_train_1k.test.jsonl --num_samples 1000 --max_length 64
└── 6) 启动服务
    └── python /root/gpufree-data/e-intent-detect-model/app.py
```

## 5. 在线服务说明（tree）

```text
FastAPI 服务
├── 健康检查
│   └── GET /healthz
├── 预测接口
│   └── POST /v1/intent/predict
│       ├── 入参: {"query":"这款适合油皮吗"}
│       └── 出参: intent_id, level1, level2, score, is_other, model_version
└── 兜底逻辑
    └── score < OTHER_THRESHOLD -> intent_id = OTHER
```

## 6. app.py 环境变量（tree）

```text
可配置项
├── MODEL_DIR        # 默认: outputs/smoke_model
├── ONNX_PATH        # 默认: ${MODEL_DIR}/model.onnx
├── INTENT_PATH      # 默认: intent/common.json
├── MAX_LENGTH       # 默认: 64
├── OTHER_THRESHOLD  # 默认: 自动阈值（未设置环境变量时）
├── MODEL_VERSION    # 默认: 模型目录名
├── HOST             # 默认: 0.0.0.0
└── PORT             # 默认: 8080
```

## 7. 已知问题与约束（tree）

```text
兼容性与约束
├── ONNX Runtime
│   ├── 不建议: onnxruntime-gpu 1.19.x（可能要求 CUDA 12）
│   └── 推荐: onnxruntime-gpu==1.16.3（CUDA 11.8）
├── 模型源
│   ├── 默认: ModelScope
│   └── 原因: 部分环境无法访问 HuggingFace
├── 训练保存
│   └── smoke_train.py 已设置 save_safetensors=False（规避 non-contiguous tensor 保存错误）
└── 性能默认参数
    ├── batch_size=1
    └── max_length<=64
```

## 8. 后续 Agent 修改规则（tree）

```text
修改规则
├── 必须保持
│   ├── 单意图输出 + OTHER 兜底
│   ├── ModelScope 可用路径
│   └── 低延迟默认参数（batch_size=1, max_length<=64）
├── 升级依赖后必须回归
│   ├── python /root/gpufree-data/e-intent-detect-model/scripts/verify_gpu.py
│   ├── python /root/gpufree-data/e-intent-detect-model/scripts/benchmark_inference.py ...
│   └── python /root/gpufree-data/e-intent-detect-model/scripts/check_onnx_consistency.py ...
└── 上线前至少验证
    ├── P95 延迟
    ├── ONNX 一致性
    └── hard_negatives/confusion_set 误判情况
```

## 9. 项目路径
- 代码编写目录：/Users/duyx/workspace/JSAI/e-intent-detect-model
- 代码运行目录：/root/gpufree-data/e-intent-detect-model
