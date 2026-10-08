"""Optional CPU-only LightGlue backend. Never consumes SAM GPU memory."""
import sys
from pathlib import Path
import cv2
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=1)
def _models():
    from lightglue import SuperPoint, LightGlue
    import torch
    torch.set_num_threads(4)
    return (SuperPoint(max_num_keypoints=1024).eval().cpu(),
            LightGlue(features='superpoint', flash=False).eval().cpu())


def detect_lightglue(reference, moving):
    deps = Path(__file__).resolve().parent / '.optional_deps'
    if str(deps) not in sys.path:
        sys.path.insert(0, str(deps))
    try:
        import torch
        from lightglue import SuperPoint, LightGlue
        from lightglue.utils import rbd
    except ImportError as exc:
        raise RuntimeError('LightGlue可选依赖未就绪；请查看README安装说明，不会冒充增强配准成功。') from exc
    from registration_pipeline import normalize_registration_image
    # CPU only: independent SAM service may still own a GPU-resident model.
    with torch.inference_mode():
        extractor, matcher = _models()
        def extract(im):
            im = normalize_registration_image(im)
            tensor = torch.from_numpy(np.ascontiguousarray(im)).float()[None]/255
            return extractor.extract(tensor, resize=1024)
        ref, mov = extract(reference), extract(moving)
        matches = rbd(matcher({'image0':mov,'image1':ref}))['matches'].cpu().numpy()
        rp = rbd(ref)['keypoints'].cpu().numpy()
        mp = rbd(mov)['keypoints'].cpu().numpy()
    return ('SuperPoint+LightGlue-CPU',
            [cv2.KeyPoint(float(x),float(y),1) for x,y in rp],
            [cv2.KeyPoint(float(x),float(y),1) for x,y in mp],
            [cv2.DMatch(int(i),int(j),0) for i,j in matches])
