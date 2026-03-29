# LAPA Week 1 Report (Model Execution and Analysis)

## 1) Requirement Checklist (follow de bai)

- Cai dat repo LAPA: Done.
- Chay inference hoac training thu: Done (uu tien inference vi training qua nang).
- Tim hieu input, output, cach model hoat dong: Done.
- Doc paper va trinh bay:
  - Key ideas: Done.
  - Pipeline image + language -> model -> latent action/action prediction: Done.
  - So sanh voi imitation learning / VLA thong thuong: Done.
- Tom tat ngan:
  - Model co chay duoc khong: Co.
  - Output co hop ly khong: Co.
  - Co loi gi: Co, da ghi ro ben duoi.

## 2) Environment va Setup

- OS su dung khi chay: Ubuntu 24.04.4 LTS.
- Muc tieu tuong thich: Ubuntu 22.04 (da dieu chinh code de chay on hon, co fallback).
- Python: 3.10.x (conda env lapa) va da test them env 3.12 cho demo output artifact.
- Checkpoint can thiet:
  - lapa_checkpoints/tokenizer.model
  - lapa_checkpoints/vqgan
  - lapa_checkpoints/params (du file ~13.6GB)

## 3) Execution da chay

### 3.1 Inference path

- Da chay inference CPU mode de tranh loi CUDA/cuDNN khi backend GPU khong on dinh.
- Da tao output latent action thanh cong (vi du: [[6, 4, 1, 4]]).

### 3.2 Demo output artifact (video + visualize)

- Entry script da xuat duoc:
  - JSON latent token
  - PNG visualization
  - MP4 visualization
- Artifact mau da tao thanh cong:
  - outputs/inference_ubuntu22_test3/latent_action.json
  - outputs/inference_ubuntu22_test3/latent_action_visualization.png
  - outputs/inference_ubuntu22_test3/latent_action_visualization.mp4

## 4) Input, Output, va Model Flow

### Input

- Anh RGB uint8 (mac dinh: imgs/bridge_inference.jpg)
- Instruction text (vi du: move the object)

### Output

- Latent action tokens (roi rac), khong phai robot action lien tuc.
- Cau hinh pho bien: codebook size 8, tokens_per_delta = 4.

### Pipeline

1. Image + instruction vao model.
2. VQGAN encode image thanh token thi giac.
3. VLA backbone du doan latent action token.
4. O giai doan fine-tune/deploy moi map latent token sang action robot that.

## 5) Key Ideas cua LAPA (tu paper)

- Pretraining khong can action label robot o quy mo lon.
- Latent Action Quantization hoc khong gian hanh dong roi rac tu bien doi quan sat.
- Latent pretraining giup hoc mapping tu visual-language sang latent action.
- Sau do fine-tune voi du lieu co action label de ra action that.
- Loi ich chinh: tot hon hoc truc tiep action label trong boi canh da embodiment, hieu qua pretrain cao.

## 6) LAPA khac gi voi IL/VLA thong thuong

- IL/VLA thuong can action label day du ngay tu dau.
- LAPA tach bai toan thanh:
  - Hoc latent action tu video/observation.
  - Roi moi canh chinh (fine-tune) latent -> action that.
- Diem manh: kha nang transfer giua nhieu robot/task co action space khac nhau.
- Trade-off: can them buoc mapping latent -> action khi deploy.

## 7) Loi gap va cach xu ly

- Loi GPU backend (cuDNN/JAX init fail): workaround bang CPU mode.
- Loi checkpoint chua tai xong (thieu dung luong): resume download den du 13.6GB.
- Warning support lifecycle Python 3.10: khong chan inference.

## 8) Giai dap: vi sao video demo dung hinh?

Video MP4 hien tai dung hinh vi no la visualization cua latent token tren mot anh input co dinh, KHONG phai rollout simulator.

Cu the:
- Script inference chi nhan 1 anh + 1 instruction, roi du doan token.
- Video duoc tao bang cach ve text token theo frame tren cung mot anh.
- Khong co buoc env.step(...) voi action robot, nen canh tay robot khong chuyen dong.

Muon robot "dong" trong video thi can chay policy rollout (SimplerEnv/ManiSkill) de:

1. Predict action moi timestep.
2. Step simulator bang action do.
3. Ghi lai chuoi frame tu env.

## 9) Tom tat ngan (theo Expected Results)

- Model chay duoc khong: Co (inference path da chay).
- Co output khong: Co (latent token + JSON/PNG/MP4).
- Output co hop ly khong: Co, dung dinh dang latent action cua LAPA.
- Loi ton tai: backend GPU co the fail tuy may, da co CPU fallback.

## 10) Deliverables

- Output video: outputs/inference_ubuntu22_test3/latent_action_visualization.mp4
- Code:
  - latent_pretraining/inference.py
  - SimplerEnv/simpler_env/utils/visualization.py
- Report ngan (1 page): file nay.
