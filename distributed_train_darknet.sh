pip install timm tlt
pip install torch==1.8.1 torchvision==0.9.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
cd /workspace/mnt/storage/songqinglong/code/project/volo

python3 -m torch.distributed.launch --nproc_per_node=8 main.py /workspace/mnt/storage/songqinglong/imagenet/ImageNet-pytorch \
  --model darknet37 --img-size 224 \
  -b 64 --lr 1.6e-3 --drop-path 0.1 --apex-amp \
  --token-label \
  --token-label-data /workspace/mnt/storage/songqinglong/imagenet/label_top5_train_nfnet