# pip install timm tlt
# pip install torch==1.8.1 torchvision==0.9.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
# cd /workspace/mnt/storage/songqinglong/code/project/volo
./distributed_train.sh 1 /workspace/mnt/storage/songqinglong/imagenet/ImageNet-pytorch \
  --model csppeleenet --img-size 224 \
  -b 40 --lr 1.6e-3 --drop-path 0.1 --apex-amp \
  --token-label --token-label-size 7 \
  --token-label-data /workspace/mnt/storage/songqinglong/imagenet/label_top5_train_nfnet

# CUDA_VISIBLE_DEVICES=0 ./distributed_train.sh 1 /workspace/mnt/storage/songqinglong/imagenet/ImageNet-pytorch \
#   --model volo_d1 --img-size 224 \
#   -b 48 --lr 1.6e-3 --drop-path 0.1 --apex-amp \
#   --token-label --token-label-size 14 \
#   --token-label-data /workspace/mnt/storage/songqinglong/imagenet/label_top5_train_nfnet