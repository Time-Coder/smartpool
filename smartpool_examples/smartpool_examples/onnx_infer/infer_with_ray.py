import os
import sys
from typing import List

import ray
from progress_info import ProgressInfo

from smartpool import DataSize


@ray.remote(num_cpus=1, memory=300 * DataSize.MB)
def infer_one(model_path: str, image_path: str, output_dir: str, module_dir: str):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    import onnxruntime as ort
    from inference import postprocess, preprocess

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    img, blob, scale, pad = preprocess(image_path)
    outputs = session.run(None, {input_name: blob})[0]
    output_path = output_dir + "/" + os.path.basename(image_path)
    postprocess(img, outputs, output_path, scale, pad)
    return output_path


def infer_with_ray(model_path: str, image_paths: List[str], output_dir: str, progress_info: ProgressInfo):
    module_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    all_futures = [infer_one.remote(model_path, p, output_dir, module_dir) for p in image_paths]

    for _ in image_paths:
        progress_info.start_one_preprocess()
        progress_info.start_one_infer()
        progress_info.start_one_postprocess()

    pending = set(all_futures)
    while pending:
        done, pending = ray.wait(list(pending), num_returns=min(10, len(pending)))
        pending = set(pending)
        for d in done:
            ray.get(d)
        for _ in range(len(done)):
            progress_info.finish_one_preprocess()
            progress_info.finish_one_infer()
            progress_info.finish_one_postprocess()

    ray.shutdown()
