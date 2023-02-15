from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector
import os
import cv2

config = '/home/toni/mmdetFlaskCrack/configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py'
checkpoint = '/home/toni/mmdetFlaskCrack/work_dirs/mask2former_res50_845/iter_12126_512_280.pth'
device='cuda:0'

model = init_detector(config, checkpoint, device=device)

def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.5,
                       title='result',
                       wait_time=0,
                       out_dir = None,
                       remove=False):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    if hasattr(model, 'module'):
        model = model.module
    out = model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=False,
        wait_time=wait_time,
        win_name=title,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
        out_dir = out_dir,
        remove = remove
    )
    return out

def test_pic(img_path):
    save_name = os.path.join('/home/toni/TinyWebServer-raw_version/root',os.path.basename(img_path))
    print(save_name)
    result = inference_detector(model, img_path)
    out = show_result_pyplot(model, img_path, result, 0.5)
    cv2.imwrite(save_name,out)
    

parser = ArgumentParser()
parser.add_argument('img_path', help='Image file')
args = parser.parse_args()

if __name__=='__main__':
    test_pic(img_path = args.img_path)