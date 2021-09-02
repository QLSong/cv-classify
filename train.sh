pip install timm tlt
pip install torch==1.8.1 torchvision==0.9.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
cd /workspace/mnt/storage/songqinglong/code/project/volo
CUDA_VISIBLE_DEVICES=0,1 ./distributed_train.sh 2 /workspace/mnt/storage/songqinglong/imagenet/ImageNet-pytorch \
  --model mobilenet_v3_small --img-size 224 \
  -b 40 --lr 1.6e-3 --drop-path 0.1 --apex-amp \
  --token-label \
  --token-label-data /workspace/mnt/storage/songqinglong/imagenet/label_top5_train_nfnet