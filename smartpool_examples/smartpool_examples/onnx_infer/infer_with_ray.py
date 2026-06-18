from typing import List
import os

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import ray

from smartpool_examples.onnx_infer.progress_info import ProgressInfo
from smartpool_examples.onnx_infer.inference import preprocess, postprocess


@ray.remote(num_gpus=0)
def ray_preprocess(image_path: str):
    return preprocess(image_path)


_session_cache = {}


def _get_session(model_path: str):
    if model_path not in _session_cache:
        import onnxruntime as ort
        _session_cache[model_path] = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
    return _session_cache[model_path]


@ray.remote(num_gpus=0)
def ray_infer(model_path: str, preprocess_result):
    session = _get_session(model_path)
    _, blob, _, _ = preprocess_result
    return session.run(None, input_feed={session.get_inputs()[0].name: blob})[0]


@ray.remote(num_gpus=0)
def ray_postprocess(preprocess_result, infer_result, output_path):
    img, _, scale, pad = preprocess_result
    postprocess(img, infer_result, output_path, scale, pad)


def infer_with_ray(model_path: str, image_paths: List[str], output_dir: str, progress_info: ProgressInfo):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    pre_refs = []
    infer_refs = []
    post_refs = []

    for i, image_path in enumerate(image_paths):
        output_path = output_dir + "/" + os.path.basename(image_path)

        progress_info.start_one_preprocess()
        pre_ref = ray_preprocess.remote(image_path)
        pre_refs.append(pre_ref)

        infer_ref = ray_infer.remote(model_path, pre_ref)
        infer_refs.append(infer_ref)

        post_ref = ray_postprocess.remote(pre_ref, infer_ref, output_path)
        post_refs.append(post_ref)

    remaining = list(pre_refs)
    while remaining:
        done, remaining = ray.wait(remaining)
        for _ in done:
            progress_info.finish_one_preprocess()
            progress_info.start_one_infer()

    remaining = list(infer_refs)
    while remaining:
        done, remaining = ray.wait(remaining)
        for _ in done:
            progress_info.finish_one_infer()
            progress_info.start_one_postprocess()

    remaining = list(post_refs)
    while remaining:
        done, remaining = ray.wait(remaining)
        for _ in done:
            progress_info.finish_one_postprocess()

    ray.get(post_refs)
