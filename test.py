import imageio
from part_swap import load_checkpoints
from skimage.transform import resize
import warnings
from skimage import img_as_ubyte
from part_swap import make_video
from utils import *
from mtcnn.mtcnn import MTCNN
from argparse import ArgumentParser
import cv2
import numpy as np
from tqdm import tqdm
from pixellib.semantic import semantic_segmentation
import time

segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("weights/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

mtcnn = MTCNN()

warnings.filterwarnings("ignore")

reconstruction_module, segmentation_module = load_checkpoints(config='config/vox-256-sem-5segments.yaml',
                                               checkpoint='weights/vox-5segments.pth.tar',
                                               blend_scale=1)
def crop_fake(source_image):
    w_,h_,_ = img2.shape
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    _, boxes = crop(mtcnn, source_image)
    boxes_ = np.array(boxes)
    boxes = boxes[np.argmax(boxes_[:, 3])]
    size = max(boxes[3], boxes[2])
    dis = boxes[3] - boxes[2]
    margin = int(size * 0.3)
    x, y, w, h = max(0,boxes[0] - margin - int(dis/2)), max(0, boxes[1] - margin), min(w_, size + 2*margin), min(h_,size + 2*margin)
    return x, y, w, h

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--outvid")
    parser.add_argument("--img" )
    parser.add_argument("--invid" )

    opt = parser.parse_args()
    s = time.time()

    # Thay thế ảnh và video bạn muốn swap, một vài ảnh và video mẫu trong thư mục data
    source_image = imageio.imread(opt.img)
    target_video = imageio.mimread(opt.invid, memtest=False)

    img2 = source_image.copy()
    x, y, w, h = crop_fake(source_image)
    source_image = img2[y: y + h, x: x + w]
    source_image = resize(source_image, (256, 256))[..., :3]
    w_s, h_s, _ = target_video[0].shape
    x, y, w, h = crop_fake(target_video[0])


    target_image = []
    for img2 in target_video:
      target_img = img2[y : y + h, x : x + w]
      target_image.append(target_img)
    target_image = [resize(frame, (256, 256))[..., :3] for frame in target_image]
    pre = make_video(swap_index=[1,2], source_image = source_image, target_video = target_image, use_source_segmentation=True,
                              segmentation_module=segmentation_module, reconstruction_module=reconstruction_module)

    scale = 256/w
    if scale < 0.7:
        dim = (int(scale * w_s), int(scale * h_s))
        x, y, w, h = int(scale*x), int(scale*y), 256, 256
        target_video = [resize(frame, dim)[..., :3] for frame in target_video]
    else:
        target_video = [frame.astype(np.float32) / 255. for frame in target_video]
        pre = [resize(frame, (w, h))[..., :3] for frame in pre]
    lb, _ = segment_image.segmentAsPascalvoc((pre[0] * 255).astype(np.uint8), process_frame=True)
    ind = np.where(lb['masks'] == True)

    for i, target in enumerate(target_video):
        tg_small = target[y: y + h, x: x + w]
        for a in range(len(ind[0])):
            tg_small[ind[0][a], ind[1][a]] = pre[i][ind[0][a], ind[1][a]]
        target[y: y + h, x: x + w] = tg_small
    print(time.time() - s)
    imageio.mimsave(opt.outvid, [img_as_ubyte(frame) for frame in target_video], fps=30)
