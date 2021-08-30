# python train.py --config=yolact_plus_resnet50_config \
# --resume=weights/yolact_plus_resnet50_29_429856_interrupt.pth \
# --start_iter=429856

python eval.py --config=yolact_plus_resnet50_config \
--trained_model=weights/yolact_plus_resnet50_32_480000.pth \
--output_coco_json \
--dataset=coco2017_testdev_dataset

# python run_coco_eval.py --eval_type=mask