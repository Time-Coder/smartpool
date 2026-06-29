from typing import List
import os
from concurrent.futures import as_completed

from progress_info import ProgressInfo
from inference import preprocess, postprocess

from smartpool import InferSessionPool, ThreadPool, Resource, DataSize

preprocess = ThreadPool.task(
    pool_name="pre_post",
    cpu_mode_res=Resource(
        cpu_cores=1,
        cpu_mem=15*DataSize.MB
    ),
    check_args=False
)(preprocess)

postprocess = ThreadPool.task(
    pool_name="pre_post",
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
        cpu_mem=81*DataSize.MB
    )
    gpu_res = Resource(
        cpu_cores_in_python=0,
        cpu_cores_out_of_python=1,
        cpu_mem=128*DataSize.MB,
        gpu_cores=100,
        gpu_mem=100*DataSize.MB
    )

    # ThreadPool.config(pool_name="pre_post", max_workers=4)
    infer_session_pool = InferSessionPool()

    futures = []
    for image_path in image_paths:
        preprocess_future = preprocess(image_path)
        preprocess_future.add_start_callback(progress_info.start_one_preprocess)
        preprocess_future.add_done_callback(progress_info.finish_one_preprocess)

        img, blob, scale, pad = preprocess_future.unpack(4)
        infer_future = infer_session_pool.submit(model_path, args=(blob,), cpu_mode_res=cpu_res, gpu_mode_res=gpu_res, use_io_binding=False, check_args=False, chunksize=2)
        infer_future.add_start_callback(progress_info.start_one_infer)
        infer_future.add_done_callback(progress_info.finish_one_infer)

        outputs = infer_future[0]
        output_path = output_dir + "/" + os.path.basename(image_path)
        postprocess_future = postprocess(img, outputs, output_path, scale, pad)
        postprocess_future.add_start_callback(progress_info.start_one_postprocess)
        postprocess_future.add_done_callback(progress_info.finish_one_postprocess)
        
        futures.append(postprocess_future)

    for future in as_completed(futures):
        future.result()