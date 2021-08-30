
# 训练脚本
# validation_epoch 用于设置 val 测试起始 epoch，默认 35W ，也就是 35w epoch 之后 每 1w 个 epoch 进行 val
# 这里设置为 10000，就是 1w 个 epoch 开始进行 valid
python train.py --config=yolact_plus_resnet50_config --batch_size=8 --validation_epoch=10000


# resume 脚本 1
# start_iter 用于加载 optimizer 的参数和 epoch，如果 start_iter 不设置，默认 -1 从 0 开始
python train.py --config=yolact_plus_resnet50_config --batch_size=8  --resume=./weights/yolact_plus_resnet50_29_429856_interrupt.dpparams --start_iter=429856

# resume 脚本 2
# python train.py --config=yolact_plus_resnet50_config \
# --resume=weights/yolact_plus_resnet50_29_429856_interrupt.pth \
# --start_iter=429856

# valid 脚本
# dataset 默认 coco2017 val2017 4952 个样本
python eval.py --config=yolact_plus_resnet50_config \
--trained_model=weights/yolact_plus_resnet50_32_480000.pth

# 预测 test-dev 
python eval.py --trained_model yolact_plus_resnet50_32_480000.dpparams --output_coco_json  --dataset=coco2017_testdev_dataset --cuda=True

# 打包预测结果
cd results

cp mask_detections.json detections_test-dev2017_yolact_results.json

zip detections_test-dev2017_yolact_results.zip detections_test-dev2017_yolact_results.json


