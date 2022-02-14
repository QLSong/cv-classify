# python validate.py /workspace/mnt/storage/songqinglong/data/ImageNet-pytorch  --model peleenet \
#   --checkpoint output/train/20210902-142702-peleenet-224/model_best.pth.tar --no-test-pool --img-size 224 -b 128

# python validate.py /workspace/mnt/storage/songqinglong/data/ImageNet-pytorch  --model SReT_T_distill \
#   --pretrained --no-test-pool --img-size 224 -b 128

python validate.py /workspace/mnt/storage/songqinglong/data/ImageNet-pytorch  --model rexnet \
  --checkpoint /workspace/mnt/storage/songqinglong/song/project/Deep-Homography-Estimation-Pytorch-master/truncation_car/rexnetv1_1.0.pth --no-test-pool --img-size 224 -b 128