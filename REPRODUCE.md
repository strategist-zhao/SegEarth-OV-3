# SegEarth-OV3 复现指南（AutoDL）

> 本人亲测可用的复现记录。目标环境：Python 3.10 + PyTorch 2.4.1 + CUDA 12.1 + mmcv 2.2.0 + mmsegmentation 1.2.2
> 基准结果：OpenEarthMap val mIoU = 44.15（论文 42.9，±1 正常）

## 0. 开实例

AutoDL 镜像选 **Miniconda / conda3 / 3.10(ubuntu22.04) / 11.8**。
（不要选现成 PyTorch 镜像：>=2.5 没有 mmcv 预编译包，<=2.3 跑不了 SAM3，2.4.x 是唯一甜点区）

## 1. 建环境

```bash
conda create -n segearth python=3.10 -y
conda activate segearth
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. PyTorch（锁定 2.4.1 + cu121）

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3. mmcv / mmsegmentation（⚠️ 关键步骤）

```bash
# mmcv 必须走 openmmlab 预编译索引，从普通源装会编译失败
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install "mmengine<1.0.0" mmsegmentation==1.2.2

# mmseg 1.2.2 版本断言 mmcv<2.2.0（不含），必须放宽一行上限：
sed -i "s/MMCV_MAX = '2.2.0'/MMCV_MAX = '2.4.0'/" /root/miniconda3/envs/segearth/lib/python3.10/site-packages/mmseg/__init__.py
```

## 4. 其余依赖

⚠️ numpy 必须 <2（SAM3 要求）；opencv 5.x 要求 numpy>=2，必须锁 4.10.0；plyfile 锁 1.0.3。

```bash
pip install numpy==1.26.4 "opencv-python==4.10.0.84" "opencv-python-headless==4.10.0.84" "timm>=1.0.17" ftfy==6.1.1 regex iopath huggingface_hub tqdm einops pycocotools psutil matplotlib scikit-image "plyfile==1.0.3" scipy openpyxl pandas

# 验证（期望输出 1.26.4 4.10.0 2.2.0 1.2.2 0.10.x）
python -c "import numpy,cv2,mmcv,mmseg,mmengine;print(numpy.__version__,cv2.__version__,mmcv.__version__,mmseg.__version__,mmengine.__version__)"
```

也可以 pip install -r requirements.txt，但必须先完成第 3 步。

## 5. 代码与 SAM3 权重

```bash
cd /root/autodl-tmp
git clone https://github.com/strategist-zhao/SegEarth-OV-3.git
cd SegEarth-OV-3
mkdir -p weights/sam3
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('facebook/sam3', local_dir='weights/sam3', allow_patterns=['*.pt'])"
ls weights/sam3   # 应为 sam3.pt（约 3.3G）
```

## 6. 数据集（OpenEarthMap 示例）

```bash
# 本地下载 OpenEarthMap_wo_xBD.zip 传到服务器 data/ 下，然后：
cd /root/autodl-tmp/SegEarth-OV-3/data
unzip OpenEarthMap_wo_xBD.zip -d OpenEarthMap_raw

cd /root/autodl-tmp/SegEarth-OV-3
curl -O https://raw.githubusercontent.com/likyoo/SegEarth-OV/main/tools/dataset_converters/openearthmap.py
mkdir -p tools/dataset_converters && mv openearthmap.py tools/dataset_converters/
python tools/dataset_converters/openearthmap.py /root/autodl-tmp/SegEarth-OV-3/data/OpenEarthMap_raw/OpenEarthMap_wo_xBD

# 转换输出是 img_dir/val + ann_dir/val，需对齐为 val/images + val/labels：
cd data/OpenEarthMap
mkdir val && mv img_dir/val val/images && mv ann_dir/val val/labels && rmdir img_dir ann_dir
ls val/images | wc -l    # 384
ls val/labels | wc -l    # 384
```

## 7. 运行（需开机有 GPU）

```bash
conda activate segearth
cd /root/autodl-tmp/SegEarth-OV-3
python demo.py                          # 单张图可视化，输出 seg_pred.png
python eval.py ./configs/cfg_openearthmap.py   # 评测，预期 mIoU 约 42.9~44.2
```

长时间评测建议用 screen 挂后台防断连。

## 已知坑

- 阿里云 pip 源缺包（addict、mmsegmentation、numpy<2 都装不到），用清华源
- zenodo 在 AutoDL 被 DNS 污染，配 source /etc/network_turbo 或本地下载后上传
- numpy 会被其它包顶回 2.x，装完务必再确认 numpy.__version__ 是 1.26.4
- 无卡模式不能跑推理，但下载/解压/转换/commit-push 都可以
