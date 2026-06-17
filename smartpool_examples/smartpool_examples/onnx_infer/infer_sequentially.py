from typing import List
import os
import onnxruntime as ort

from progress_info import ProgressInfo
from inference import preprocess, postprocess
from smartpool import TRT_CACHE_PATH


def infer_sequentially(model_path: str, image_paths: List[str], output_dir: str, progress_info: ProgressInfo):
    providers = ort.get_available_providers()
    for i in range(len(providers)):
        if providers[i] == "TensorrtExecutionProvider":
            providers[i] = ("TensorrtExecutionProvider", {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": TRT_CACHE_PATH
            })
    model: ort.InferenceSession = ort.InferenceSession(model_path, providers=providers)
    input_name: str = model.get_inputs()[0].name

    for image_path in image_paths:
        progress_info.start_one_preprocess()
        img, blob, scale, pad = preprocess(image_path)
        progress_info.finish_one_preprocess()

        progress_info.start_one_infer()
        outputs = model.run(None, input_feed={input_name: blob})[0]
        progress_info.finish_one_infer()

        output_path = output_dir + "/" + os.path.basename(image_path)
        progress_info.start_one_postprocess()
        postprocess(img, outputs, output_path, scale, pad)
        progress_info.finish_one_postprocess()
        