import os
from concurrent.futures import as_completed
from typing import List

from inference import postprocess, preprocess
from progress_info import ProgressInfo

from smartpool import DataSize, InferSessionPool, Resource, ThreadPool

preprocess = ThreadPool.task(
    cpu_mode_res=Resource(
        cpu_cores=1,
        cpu_mem=15*DataSize.MB,
        result_cpu_mem=6*DataSize.MB
    ),
    check_args=False
)(preprocess)

postprocess = ThreadPool.task(
    cpu_mode_res=Resource(
        cpu_cores=1,
        cpu_mem=81*DataSize.MB
    ),
    check_args=False
)(postprocess)


def infer_with_smartpool(model_path: str, image_paths: List[str], output_dir: str, progress_info: ProgressInfo):
    cpu_res = Resource(
        cpu_cores_in_python=0,
        cpu_cores_out_of_python=1,
        cpu_mem=81*DataSize.MB,
        result_cpu_mem=3*DataSize.MB
    )
    gpu_res = Resource(
        cpu_cores_in_python=0,
        cpu_cores_out_of_python=1,
        cpu_mem=128*DataSize.MB,
        gpu_cores=100,
        gpu_mem=10*DataSize.MB,
        result_cpu_mem=3*DataSize.MB,
        result_gpu_mem=3*DataSize.MB
    )

    infer_model = InferSessionPool.model(
        model_path,
        cpu_mode_res=cpu_res,
        gpu_mode_res=gpu_res,
        use_io_binding=False,
        check_args=False,
        chunksize=1
    )

    futures = []
    for image_path in image_paths:
        preprocess_future = preprocess(image_path)
        preprocess_future.add_start_callback(progress_info.start_one_preprocess)
        preprocess_future.add_done_callback(progress_info.finish_one_preprocess)
        # preprocess_future.result()

        img, blob, scale, pad = preprocess_future.unpack(4)
        infer_future = infer_model(blob)
        infer_future.add_start_callback(progress_info.start_one_infer)
        infer_future.add_done_callback(progress_info.finish_one_infer)
        # infer_future.result()

        outputs = infer_future[0]
        output_path = output_dir + "/" + os.path.basename(image_path)
        postprocess_future = postprocess(img, outputs, output_path, scale, pad)
        postprocess_future.add_start_callback(progress_info.start_one_postprocess)
        postprocess_future.add_done_callback(progress_info.finish_one_postprocess)
        # postprocess_future.result()

    for future in futures:
        future.result()
