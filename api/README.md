# AEOT API (Single GPU)

这个目录提供一个最小可用的网页后端方案：
- 前端点击后调用 `POST /generate`
- 后端把任务放进队列
- 单 worker 串行执行 `scripts/run_aeot_end2end.py`
- 固定占用同一张 GPU（默认 `0`）

## 1. 启动

```bash
cd /home/zky/PyTorch-VAE
export CUDA_VISIBLE_DEVICES=0
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

## 2. 发起任务

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "ae_ckpt": "/home/zky/PyTorch-VAE/checkpoints/aeot_sigmoid/your.ckpt",
    "n_generate": 1000,
    "num_gen_x": 50000,
    "seed": 42
  }'
```

返回示例：

```json
{
  "task_id": "3a2c7e9b1f4d",
  "status": "queued",
  "queue_size": 1
}
```

## 3. 查询任务

```bash
curl http://127.0.0.1:8000/tasks/3a2c7e9b1f4d
```

完成后可从返回中拿到：
- `run_dir`
- `summary_path`
- `summary.outputs.filtered_dir`

## 4. 说明
- 该实现是单机单卡串行模板，适合你当前“网页点击一键生成”的阶段。
- 如果后面并发上来，再扩展为：
  - API 只负责入队
  - Redis 队列 + 独立 worker
  - 按 GPU 分片多个 worker
