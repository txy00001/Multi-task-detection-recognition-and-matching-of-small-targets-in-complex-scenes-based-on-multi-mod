import time
import fastdeploy as fd
import cv2
import os
import argparse
import numpy as np

args = argparse.Namespace(
    det_model="demo/model/new_ocr/det",
    rec_model="demo/model/new_ocr/rec",
    rec_label_file="demo/model/txt/qx_ocr_dict.txt",
    image='demo/pic/6.jpg',
    device="gpu",
    device_id=1,
    rec_bs=16,
    backend="trt",
)

def build_option(args):
    det_option = fd.RuntimeOption()
    rec_option = fd.RuntimeOption()

    det_option.trt_option.enable_fp16 = True
    rec_option.trt_option.enable_fp16 = True
    
    det_option.use_gpu(args.device_id)
    rec_option.use_gpu(args.device_id)

    if args.backend.lower() == "trt":
        assert args.device.lower() == "gpu", "TensorRT backend require inference on device GPU."
        det_option.use_trt_backend()
        rec_option.use_trt_backend()
          # 添加动态shape
        det_option.trt_option.set_shape("x", [1, 3, 64, 64],
                                     [1, 3, 960, 960],
                                     [1, 3, 3840, 3840],)
        rec_option.trt_option.set_shape("x", [1, 3, 48, 10],
                                     [args.rec_bs, 3, 48, 320],
                                     [args.rec_bs, 3, 48, 2304],)

        # 添加序列化文件
        det_option.trt_option.serialize_file = f"{args.det_model}/det_trt_cache_16_new.trt"
        rec_option.trt_option.serialize_file = f"{args.rec_model}/rec_trt_cache_16_new.trt"
        
        
        
    elif args.backend.lower() == "pptrt":
        assert args.device.lower(
        ) == "gpu", "Paddle-TensorRT backend require inference on device GPU."
        det_option.use_paddle_infer_backend()
        det_option.paddle_infer_option.collect_trt_shape = True
        det_option.paddle_infer_option.enable_trt = True

        rec_option.use_paddle_infer_backend()
        rec_option.paddle_infer_option.collect_trt_shape = True
        rec_option.paddle_infer_option.enable_trt = True

        det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 960, 960],
                                       [1, 3, 2160, 3840])
        
        rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                       [args.rec_bs, 3, 48, 320],
                                       [args.rec_bs, 3, 48, 2304])

        # Users could save TRT cache file to disk as follow.
        det_option.trt_option.serialize_file = f"{args.det_model}/det_pptrt_cache_16.trt"
        rec_option.trt_option.serialize_file = f"{args.rec_model}/rec_pptrt_cache_16.trt"
        
    return det_option, rec_option

det_model_file = os.path.join(args.det_model, "inference.pdmodel")
det_params_file = os.path.join(args.det_model, "inference.pdiparams")
rec_model_file = os.path.join(args.rec_model, "inference.pdmodel")
rec_params_file = os.path.join(args.rec_model, "inference.pdiparams")
rec_label_file = args.rec_label_file

det_option, rec_option = build_option(args)
det_option.trt_option.enable_fp16 = True
rec_option.trt_option.enable_fp16 = True

det_model = fd.vision.ocr.DBDetector(det_model_file, det_params_file, runtime_option=det_option)
rec_model = fd.vision.ocr.Recognizer(rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)

det_model.preprocessor.max_side_len = 3840
det_model.postprocessor.det_db_thresh = 0.2 ##得分大于该阈值的像素点才会被认为是文字像素点
det_model.postprocessor.det_db_box_thresh = 0.2 ##检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域
det_model.postprocessor.det_db_unclip_ratio = 2 ##对文字区域进行扩张
det_model.postprocessor.det_db_score_mode = "slow"
det_model.postprocessor.use_dilation = True


ppocr_v4 = fd.vision.ocr.PPOCRv4(det_model=det_model, rec_model=rec_model)
ppocr_v4.rec_batch_size = args.rec_bs

# Read the input image
im = cv2.imread(args.image)

# Predict and reutrn the results
start = time.time()
result = ppocr_v4.predict(im)
end = time.time()

print("cost : {} s".format((end - start)))
print(result)

# Visuliaze the results.
vis_im = fd.vision.vis_ppocr(im, result)
cv2.imwrite("/home/txy/code/FastDeploy/demo/pic_out/result_6_trt16_new.jpg", vis_im)


