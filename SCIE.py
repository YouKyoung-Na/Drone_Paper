import argparse
import os
import platform
import sys
from pathlib import Path
import copy
import torch

## for time checking 
import cv2


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative




str_ROOT = str(ROOT)
sys.path.append('.\\detector_tracker')
sys.path.append('.\\detector_tracker\\yolov5')
sys.path.append('.\\detector_tracker\\trackers')
sys.path.append('.\\detector_tracker\\trackers\\strongsort')




from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                            increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

###### FOR TRACKING ######
from trackers.multi_tracker_zoo import create_tracker

###### FOR TRACKING ######


### setting module
# For plus module
from datetime import datetime
import natsort
import math
import numpy as np
from PIL import Image

# For threading module
import threading
import time
import logging

# 서빈표 module
from methods.dimension import Dimension

### Sub Thread execution
def work(yoloform, class_number, img, crop):
	# parameter setting
	D = Dimension(yoloform, class_number, img, crop)
	
	# find rotation information
	left_comma, cc_xy = D.rotation()
	
	# calculate detph information
	z_info, left_c, right_c = D.depth(left_comma, cc_xy, class_number)
	return z_info, left_c, right_c
	# object visualization using depth information
#	D.visualization(z_info, left_c, right_c)
    
### Main
@smart_inference_mode()
def run(
        weights=ROOT / 'best.pt',  # model path or triton URL
        source=ROOT / 'detector_tracker/data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'detector_tracker/data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name

        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        
        
        ###### FOR TRACKING ######
        reid_weights=ROOT / 'osnet_x0_25_msmt17.pt',
        tracking_method='bytetrack',
        tracking_config=None,
        tracking_result=ROOT /'runs/track',  # save results to project/name
        hide_class=False,
        save_MOT_label=True,
        ###### FOR TRACKING ######
        blur=False,



):

    ##### 속도 측정 #####
    time = 0
    cnt = 0

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download


    ###### FOR TRACKING ######
    save_track_dir = increment_path(Path(tracking_result) / name, exist_ok=exist_ok)  # increment run
    ###### FOR TRACKING ######



    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    ###### FOR TRACKING ######
    nr_sources = 1
    ###### FOR TRACKING ######
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset) # batch_size
        ###### FOR TRACKING ######
        nr_sources = len(dataset)
        ###### FOR TRACKING ######

    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    vid_path, vid_writer = [None] * bs, [None] * bs



###### FOR TRACKING ######
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources
###### FOR TRACKING ######

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    ###### FOR TRACKING ######
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    ###### FOR TRACKING ######
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset): # 데이터셋 목록에서 하나씩 이미지랑 속성 받아옴(Frame당 한번씩 반복)
        
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_track_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) # Detrction 결과가 pred에 할당 

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image, # NMS들어갔다가 나온 Detection 결과 하나하나마다 반복적용

            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                
                ###### FOR TRACKING ######
                p = Path(p)
                s += f'{i}: '
                txt_file_name = p.name
                
                ###### FOR TRACKING ######
                
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
                ###### FOR TRACKING ######
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    
                ###### FOR TRACKING ######
                
            p = Path(p)  # to Path



            ###### FOR TRACKING ######
            curr_frames[i] = im0
            ###### FOR TRACKING ######
            
            # Save path and file name with current time

            # Drone and Package detection -> False to True
            
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            ###### FOR TRACKING ######
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
            ###### FOR TRACKING ######
            
            ###### FOR TRACKING ######
            # pass detections to strongsort
            with dt[3]:
                outputs[i] = tracker_list[i].update(det.cpu(), im0)
            ###### FOR TRACKING ######
            
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()



            ###### FOR TRACKING ######
                if len(outputs[i]) > 0:
                    for j, (output) in enumerate(outputs[i]): 
                        
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                        
                        
                        
                        


                        # ###### FOR TRACKING HY ######
                        if not os.path.isdir(str(save_track_dir)+'/'+f'ID{str(id).zfill(4)}'):
                            id_dir = str(save_track_dir)+'/'+f'ID{str(id).zfill(4)}'
                            os.makedirs(id_dir)
                            id_dir_path = Path(id_dir)
                            (id_dir_path / 'labels').mkdir(parents=True, exist_ok=True)
                            (id_dir_path / 'images').mkdir(parents=True, exist_ok=True)
                            (id_dir_path / 'results').mkdir(parents=True, exist_ok=True)
                            (id_dir_path / 'crop').mkdir(parents=True, exist_ok=True)
                        else:
                            id_dir = str(save_track_dir)+'/'+f'ID{str(id).zfill(4)}'
                            id_dir_path = Path(id_dir)
                            
                        yoloform = str(int(cls)) +' '+ str((bbox[0] + (output[2] - output[0])/2)/640) +' '+ str((bbox[1] + (output[3] - output[1])/2)/480) +' '+ str((output[2] - output[0])/640) +' '+ str((output[3] - output[1])/480)
                        yoloform_additional = yoloform + ' ' + str(id) + ' ' +str(frame_idx)
                        #yolo format에 맞춰서 결과 저장하도록 만듬
                        with open(str(id_dir_path) + '/labels/' + 'yolo.txt', 'a') as f:
                                f.write(yoloform_additional + '\n')





                        
                        # to drone crop
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                        color = colors(c, True)
                        annotator.box_label(bbox, label, color=color)
                        
                        
                        if save_crop:
                            txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                            crop=save_one_box(np.array(bbox, dtype=np.int16), imc, file=id_dir_path /'crop'/ txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                        
                    
                        
                        
                        z_info, left_c, right_c = work(yoloform, cls, im0, crop)
                        save_res_form = str(z_info) + ' ' + str(left_c[0]) + ' ' + str(left_c[1])+ ' ' + str(left_c[2])+ ' ' + str(right_c[0])+ ' ' + str(right_c[1])+ ' ' + str(right_c[2])
                        
                        
                        #img save
                        if save_img:
                            cv2.imwrite(str(id_dir_path)+'/images/'+f'{frame_idx}.jpg', im0)
                        with open(str(id_dir_path) + '/results/' + 'result.txt', 'a') as f:
                                f.write(save_res_form + '\n')
                        
            else:
                pass

                        
                        
                        
                        # token_folder = natsort.natsorted(os.listdir("./"+str(save_dir)+"/labels/"))
						
                        # if (len(token_folder) > 91): # 30 frame = 3s
                        #     args_thread = token_folder[:-2]
                        #     x = threading.Thread(target=work, args=(args_thread, save_dir))
                        #     x.start() # sub thread start!
                    # if save_img or save_crop or view_img:  # Add bbox to image
                    #     c = int(cls)  # integer class
                    #     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
#                        annotator.box_label(xyxy, label, color=colors(c, True))
                    # if save_crop:
                    #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)


            # Stream results
#            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_drone: # drone!
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #             cv2.imwrite(save_path, im0) ### IMAGE SAVE!!!!!
            #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer[i].write(im0)

###### FOR TRACKING ###### 
            prev_frames[i] = curr_frames[i]

# Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

        ############## yk for time checking #################
        time += sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3
        cnt += 1
        ############## yk for time checking #################

###### FOR TRACKING ######
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


    return (time, cnt)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/DJI_DRONE.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', default=True, help='show results')
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', default=True, help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', default=True, help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', default=False, help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    
    
    ###### FOR TRACKING ######
    parser.add_argument('--reid-weights', type=Path, default=ROOT / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='bytetrack', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-result', default=ROOT / 'runs/track')
    parser.add_argument('--tracking-config', type=Path, default='.\\detector_tracker\\trackers\\bytetrack\\configs\\bytetrack.yaml')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--save-MOT-label', default=True, action='store_true', help='save label MOT fomet')
    ###### FOR TRACKING ######
    

    
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))


    return opt


def main(opt):

    check_requirements(exclude=('tensorboard', 'thop'))
    tm, cnt = run(**vars(opt))
    return tm, cnt



## yk - 시간측정 ##


if __name__ == "__main__":


    opt = parse_opt()
    tm, cnt = main(opt)
    print(f'\n 속도 측정(ms) : {tm / cnt:.3f} \n')
    ## yk - 시간측정 ##
