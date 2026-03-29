# Hướng dẫn chạy workspace này (LAPA) — Inference + Training

Repo trong workspace này chủ yếu nằm ở thư mục `LAPA/` (JAX/Flax). Ngoài ra có kèm `LAPA/SimplerEnv/` để chạy rollout/evaluation trong simulator.

> Nếu bạn chỉ cần “chạy được + có output video/visualize”, xem mục **Quickstart (Inference)**.

## Mục lục

- [1) Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
- [2) Lấy code + cấu trúc thư mục](#2-lấy-code--cấu-trúc-thư-mục)
- [3) Tạo môi trường Python (venv/conda)](#3-tạo-môi-trường-python-venvconda)
- [4) Cài dependency](#4-cài-dependency)
  - [4.1 Inference (khuyến nghị để bắt đầu)](#41-inference-khuyến-nghị-để-bắt-đầu)
  - [4.2 Full train stack](#42-full-train-stack)
  - [4.3 GPU (CUDA 12)](#43-gpu-cuda-12)
- [5) Tải checkpoints (bắt buộc cho chạy model thật)](#5-tải-checkpoints-bắt-buộc-cho-chạy-model-thật)
- [6) Quickstart: chạy inference + xuất JSON/PNG/MP4](#6-quickstart-chạy-inference--xuất-jsonpngmp4)
  - [6.1 Chạy inference “đúng model”](#61-chạy-inference-đúng-model)
  - [6.2 Chạy fallback/mock (khi thiếu JAX/GPU)](#62-chạy-fallbackmock-khi-thiếu-jaxgpu)
  - [6.3 Output tạo ra](#63-output-tạo-ra)
- [7) Fine-tune LAPA (latent → action thật)](#7-fine-tune-lapa-latent--action-thật)
  - [7.1 Chuẩn bị dữ liệu real-world](#71-chuẩn-bị-dữ-liệu-real-world)
  - [7.2 Fine-tune trên SIMPLER trajectories](#72-fine-tune-trên-simpler-trajectories)
- [8) Latent-pretraining trên Open-X (từ LWM checkpoint)](#8-latent-pretraining-trên-open-x-từ-lwm-checkpoint)
- [9) LAQ (Latent Action Quantization)](#9-laq-latent-action-quantization)
- [10) (Tuỳ chọn) SimplerEnv rollout/evaluation với policy LAPA](#10-tuỳ-chọn-simplerenv-rolloutevaluation-với-policy-lapa)
- [11) Troubleshooting (lỗi hay gặp)](#11-troubleshooting-lỗi-hay-gặp)

---

## 1) Yêu cầu hệ thống

**OS khuyến nghị**: Ubuntu 22.04 (Jammy) hoặc 24.04.

**Python**: khuyến nghị Python **3.10** (repo có thể chạy 3.12 tuỳ gói, nhưng training thường “nhạy” hơn).

**Phần cứng**

- Inference CPU: checkpoint rất lớn (~13.6GB) nên thường cần **RAM >= 16GB** để chạy ổn.
- Training: cần multi-GPU (scripts mẫu dùng 4 hoặc 8 devices). CPU training không thực tế.

**Gói hệ thống tối thiểu (Ubuntu)**

```bash
sudo apt update
sudo apt install -y git wget python3.10 python3.10-venv python3-pip ffmpeg
```

- `ffmpeg` giúp xuất MP4 (dù `imageio[ffmpeg]` có thể tự kéo binary, cài sẵn vẫn ổn định hơn).

---

## 2) Lấy code + cấu trúc thư mục

Trong workspace này:

- `LAPA/` — code chính của LAPA (inference/train)
- `LAPA/laq/` — code LAQ (huấn luyện/encode latent actions)
- `LAPA/SimplerEnv/` — simulator env + evaluation pipeline (nặng, nhiều deps)

Nếu bạn clone từ đầu (tham khảo):

```bash
git clone <your_repo_url>
cd <repo>
```

---

## 3) Tạo môi trường Python (venv/conda)

### Option A: venv (khuyến nghị)

```bash
cd LAPA
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Option B: conda

```bash
conda create -n lapa python=3.10 -y
conda activate lapa
cd LAPA
```

> Mẹo: để cache model/dataset gọn trong workspace, có thể set:
>
> ```bash
> export HF_HOME="$PWD/../.hf_home"
> ```

---

## 4) Cài dependency

Có 3 “mức” cài đặt chính:

### 4.1 Inference (khuyến nghị để bắt đầu)

```bash
cd LAPA
source .venv/bin/activate
pip install -r requirements.inference.txt
```

### 4.2 Full train stack

```bash
cd LAPA
source .venv/bin/activate
pip install -r requirements.txt
```

### 4.3 GPU (CUDA 12)

Nếu máy có CUDA 12.x và bạn muốn chạy JAX/TensorFlow GPU:

```bash
cd LAPA
source .venv/bin/activate
pip install -r requirements.gpu_cuda12.txt
```

**Lưu ý**

- `requirements.gpu_cuda12.txt` cài `jax[cuda12]==0.4.23` theo link wheel của JAX.
- Nếu gặp xung đột CUDA/cuDNN, hãy thử inference CPU trước (mục troubleshooting).

**Check nhanh dependency**

```bash
python check_deps.py
```

---

## 5) Tải checkpoints (bắt buộc cho chạy model thật)

Inference chuẩn của LAPA cần 3 file dưới `LAPA/lapa_checkpoints/`:

- `tokenizer.model`
- `vqgan`
- `params` (~13.6GB)

Tải bằng `wget`:

```bash
cd LAPA
mkdir -p lapa_checkpoints
cd lapa_checkpoints
wget -c https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/tokenizer.model
wget -c https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/vqgan
wget -c https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/params
```

Kiểm tra file `params` đủ size:

```bash
ls -lh params
```

---

## 6) Quickstart: chạy inference + xuất JSON/PNG/MP4

Entry point inference hiện tại: `python -m latent_pretraining.inference`.

### 6.1 Chạy inference “đúng model”

```bash
cd LAPA
source .venv/bin/activate

# (khuyến nghị) tránh JAX ăn hết VRAM nếu chạy GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m latent_pretraining.inference \
  --input-image imgs/bridge_inference.jpg \
  --instruction "move the object" \
  --output-dir outputs/inference_run
```

Nếu GPU backend lỗi, có thể ép CPU:

```bash
export JAX_PLATFORMS=cpu
python -m latent_pretraining.inference --output-dir outputs/inference_cpu
```

Nếu bị `Killed` hoặc `msgpack.exceptions.BufferFull`, thử giảm buffer streaming:

```bash
export LAPA_CHECKPOINT_BUFFER_GB=8
python -m latent_pretraining.inference --output-dir outputs/inference_buffer8
```

### 6.2 Chạy fallback/mock (khi thiếu JAX/GPU)

- **Mock bắt buộc** (không load model):

```bash
python -m latent_pretraining.inference --mock --output-dir outputs/inference_mock
```

- **Cho phép fallback** (thử chạy model, lỗi thì tự rơi về mock):

```bash
python -m latent_pretraining.inference --allow-mock-on-error --output-dir outputs/inference_fallback
```

> Chế độ mock/fallback chủ yếu để kiểm tra pipeline I/O + visualize + video; muốn đánh giá chất lượng latent action thật thì cần chạy model thật (mục 6.1).

### 6.3 Output tạo ra

Trong `--output-dir` sẽ có:

- `latent_action.json` — tokens + metadata (kèm `mode` và `model_error` nếu fallback)
- `latent_action_visualization.png` — ảnh visualize instruction + tokens
- `latent_action_visualization.mp4` — video overlay tokens (có thể “đứng hình” vì chỉ 1 ảnh)
- `latent_action_robot_motion_demo.mp4` — video demo “robot motion” tổng hợp theo token (không phải rollout simulator)

Tắt video nếu cần:

```bash
python -m latent_pretraining.inference --no-save-video --no-save-robot-motion-video
```

---

## 7) Fine-tune LAPA (latent → action thật)

Fine-tune dùng entrypoint `python -m latent_pretraining.train` (JAX multi-device). Repo có sẵn 2 script mẫu:

- `LAPA/scripts/finetune_real.sh`
- `LAPA/scripts/finetune_simpler.sh`

### 7.1 Chuẩn bị dữ liệu real-world

Script preprocess:

```bash
cd LAPA
source .venv/bin/activate
python data/finetune_preprocess.py \
  --input_path "/path/to/your_dataset.json" \
  --output_filename "data/real_finetune.jsonl" \
  --csv_filename "data/real_finetune.csv"
```

Sau đó chạy fine-tune:

1) Mở `LAPA/scripts/finetune_real.sh` và set biến:

- `absolute_path=/đường/dẫn/tuyệt/đối/tới/LAPA` (quan trọng)

2) Chạy:

```bash
cd LAPA
bash scripts/finetune_real.sh
```

### 7.2 Fine-tune trên SIMPLER trajectories

Dữ liệu ví dụ: `LAPA/data/simpler.jsonl`.

Tương tự, sửa `absolute_path` trong `LAPA/scripts/finetune_simpler.sh` rồi chạy:

```bash
cd LAPA
bash scripts/finetune_simpler.sh
```

**Gợi ý chỉnh để chạy ít GPU hơn**

Trong các script, `--mesh_dim='!-1,4,1,1'` nghĩa là cấu hình cho 4 devices. Nếu bạn có 1 GPU, thử đổi thành:

- `--mesh_dim='!-1,1,1,1'`

(Training vẫn nặng; đây chỉ là hướng dẫn cơ bản để khởi chạy.)

---

## 8) Latent-pretraining trên Open-X (từ LWM checkpoint)

Script: `LAPA/scripts/latent_pretrain_openx.sh`.

Cần:

- LWM checkpoint tại `LAPA/lwm_checkpoints/params`
- Dataset tại `LAPA/data/latent_action_pretraining_openx.jsonl`

Các bước:

1) Tạo thư mục:

```bash
cd LAPA
mkdir -p lwm_checkpoints data
```

2) Tải LWM checkpoint và dataset theo hướng dẫn trong `LAPA/README.md` (mục Latent-Pretraining).

3) Mở `LAPA/scripts/latent_pretrain_openx.sh` và set `absolute_path=/đường/dẫn/tuyệt/đối/tới/LAPA`.

4) Chạy:

```bash
cd LAPA
bash scripts/latent_pretrain_openx.sh
```

---

## 9) LAQ (Latent Action Quantization)

LAQ là phần học “latent action codebook” theo kiểu inverse-dynamics.

Cách chạy theo README của repo:

```bash
cd LAPA/laq
pip install -e .

# train (yêu cầu accelerate + GPU)
accelerate launch train_sthv2.py

# inference (xuất latent actions)
python inference_sthv2.py
```

---

## 10) (Tuỳ chọn) SimplerEnv rollout/evaluation với policy LAPA

Nếu bạn muốn video “robot thật sự chuyển động” theo rollout trong simulator, hãy dùng `LAPA/SimplerEnv/`.

Tóm tắt (xem chi tiết ở `LAPA/SimplerEnv/README.md`):

```bash
cd LAPA/SimplerEnv

# khuyến nghị: tạo env riêng vì deps rất nặng
conda create -n simpler_env python=3.10 -y
conda activate simpler_env

cd ManiSkill2_real2sim
pip install -e .
cd ..
pip install -e .
```

Chạy script mẫu rollout với LAPA:

```bash
cd LAPA/SimplerEnv
bash scripts/lapa_bridge.sh
```

**Lưu ý quan trọng**

- Cần GPU (SAPIEN/ManiSkill render).
- `ckpt_path="params::"` trong script cần trỏ tới checkpoint thực tế (hoặc sửa theo hướng dẫn trong code).

---

## 11) Troubleshooting (lỗi hay gặp)

### `ModuleNotFoundError: No module named 'jax'`

- Cài dependencies:

```bash
cd LAPA
source .venv/bin/activate
pip install -r requirements.inference.txt
```

- Hoặc để chạy pipeline output nhanh: dùng `--mock` / `--allow-mock-on-error`.

### GPU backend lỗi (CUDA/cuDNN/JAX init fail)

- Thử ép CPU:

```bash
export JAX_PLATFORMS=cpu
python -m latent_pretraining.inference
```

- Nếu muốn GPU: dùng `requirements.gpu_cuda12.txt` và kiểm tra CUDA driver.

### Checkpoint `params` tải thiếu → lỗi missing parameters

- Tải lại `params` bằng `wget -c ...` và kiểm tra `ls -lh`.

### `Killed` / out-of-memory khi load checkpoint

- Thử giảm buffer:

```bash
export LAPA_CHECKPOINT_BUFFER_GB=8
```

- Đảm bảo máy có đủ RAM.

### MP4 không xuất được

- Cài ffmpeg:

```bash
sudo apt install -y ffmpeg
```

- Đảm bảo đã cài `imageio[ffmpeg]`:

```bash
pip install imageio[ffmpeg]
```

---

## Tài liệu tham khảo trong repo

- `LAPA/README.md` — hướng dẫn chính của upstream
- `LAPA/LAPA_WEEK1_REPORT.md` — report 1 trang + pipeline figure
- `LAPA/latent_pretraining/inference.py` — inference entrypoint (xuất JSON/PNG/MP4)
- `LAPA/scripts/*.sh` — các lệnh train mẫu
- `LAPA/SimplerEnv/README.md` — hướng dẫn chạy simulator rollout/eval
