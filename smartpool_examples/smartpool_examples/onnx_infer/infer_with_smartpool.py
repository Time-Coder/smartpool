from .progress_info import ProgressInfo
from .inference import preprocess

def infer_with_smartpool(image_paths, progress_info: ProgressInfo):
    futures = []
    for image_path in image_paths:
        # update_task_count()
        image_path = str(image_path)
        preprocess_future = preprocess(image_path)
        preprocess_future.add_start_callback(lambda: progress_info.progress.update(progress_info.preprocess_start_task, advance=1))
        preprocess_future.add_done_callback(lambda future: progress_info.progress.update(preprocess_done_progress_task, advance=1))

        img, blob, scale, pad = preprocess_future.unpack(4)
        infer_future = InferSessionPool.run_async(model_path_str, args=(blob,), cpu_mode_res=cpu_res, gpu_mode_res=gpu_res)
        infer_future.add_start_callback(lambda: progress.update(infer_start_progress_task, advance=1))
        infer_future.add_done_callback(lambda future: progress.update(infer_done_progress_task, advance=1))

        outputs = infer_future[0]
        output_path = output_dir_str + "/" + os.path.basename(image_path)
        postprocess_future = postprocess(img, outputs, output_path, scale, pad)
        postprocess_future.add_start_callback(lambda: progress.update(postprocess_start_progress_task, advance=1))
        postprocess_future.add_done_callback(lambda future: progress.update(postprocess_done_progress_task, advance=1))
        
        futures.append(postprocess_future)

    for future in as_completed(futures):
        future.result()