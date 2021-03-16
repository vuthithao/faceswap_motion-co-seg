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
import pixellib
from pixellib.semantic import semantic_segmentation

segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

mtcnn = MTCNN()

warnings.filterwarnings("ignore")

reconstruction_module, segmentation_module = load_checkpoints(config='config/vox-256-sem-5segments.yaml',
                                               checkpoint='vox-5segments.pth.tar',
                                               blend_scale=1)
def crop_fake(source_image):
    w_,h_,_ = img2.shape
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    _, boxes = crop(mtcnn, source_image)
    if boxes:
        boxes_ = np.array(boxes)
        boxes = boxes[np.argmax(boxes_[:, 3])]
        size = max(boxes[3], boxes[2])
        dis = boxes[3] - boxes[2]
        margin = int(size * 0.3)
        x, y, w, h = max(0,boxes[0] - margin - int(dis/2)), max(0, boxes[1] - margin), min(w_, size + 2*margin), min(h_,size + 2*margin)
        return x, y, w, h, 1
    else:
        return 0,0,0,0,0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--outvid")
    parser.add_argument("--img" )
    parser.add_argument("--invid" )

    opt = parser.parse_args()

    # Thay thế ảnh và video bạn muốn swap, một vài ảnh và video mẫu trong thư mục data
    source_image = imageio.imread(opt.img)
    target_video = imageio.mimread(opt.invid, memtest=False)

    img2 = source_image.copy()
    x, y, w, h, _ = crop_fake(source_image)
    source_image = img2[y: y + h, x: x + w]
    source_image = resize(source_image, (256, 256))[..., :3]
    w_s, h_s, _ = target_video[0].shape

    target_vid = []
    for i in tqdm(target_video):
        x, y, w, h, have_face = crop_fake(i)
        if have_face:
            target_img = i[y: y + h, x: x + w]
            target_img = resize(target_img, (256, 256))[..., :3]
            pre = make_video(swap_index=[2], source_image=source_image, target_video=[target_img],
                             use_source_segmentation=True,
                             segmentation_module=segmentation_module, reconstruction_module=reconstruction_module)
            scale = 256 / w
            if scale < 0.7:
                dim = (int(scale * w_s), int(scale * h_s))
                x, y, w, h = int(scale * x), int(scale * y), 256, 256
                target = resize(i, dim)[..., :3]
            else:
                target = i.astype(np.float32) / 255.
                pre = [resize(frame, (w, h))[..., :3] for frame in pre]

            lb, _ = segment_image.segmentAsPascalvoc((pre[0]*255).astype(np.uint8), process_frame = True)
            tg_small = target[y: y + h, x: x + w]
            ind = np.where(lb['masks'] == True)
            for a in range(len(ind[0])):
                tg_small[ind[0][a], ind[1][a]] = pre[0][ind[0][a], ind[1][a]]
            target[y: y + h, x: x + w] = tg_small
            target_vid.append(target)
        else:
            target_vid.append(i)
    target_vid = [resize(frame, (w_s, h_s))[..., :3] for frame in target_vid]
    imageio.mimsave(opt.outvid, [img_as_ubyte(frame) for frame in target_vid], fps=60)
