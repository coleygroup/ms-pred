import sys
import signal
import queue
import time
import itertools
from tqdm import tqdm
import multiprocess.context as ctx
ctx._force_start_method('spawn')

def simple_parallel(
    input_list, function, max_cpu=16, timeout=4000, max_retries=3, use_ray: bool = False, task_name="",
):
    """Simple parallelization.

    Use map async and retries in case we get odd stalling behavior.

    input_list: Input list to op on
    function: Fn to apply
    max_cpu: Num cpus
    timeout: Length of timeout
    max_retries: Num times to retry this
    use_ray
    spawn=True

    """
    from pathos import multiprocessing as mp
    if spawn:
        ctx._force_start_method('spawn')

    cpus = min(mp.cpu_count(), max_cpu)
    with mp.ProcessPool(processes=cpus) as pool:
        results = list(tqdm(pool.imap(function, input_list), desc=task_name))

    return results

# # If parallel with default multi-processing, this class is needed to pass the function call object
# class batch_func:
#     def __init__(self, func, args=None, kwargs=None):
#         self.func = func
#         if args is not None:
#             self.args = args
#         else:
#             self.args = []
#         if kwargs is not None:
#             self.kwargs = kwargs
#         else:
#             self.kwargs = {}
#
#     def __call__(self, list_inputs):
#         outputs = []
#         for i in list_inputs:
#             outputs.append(self.func(i, *self.args, **self.kwargs))
#         return outputs


def chunked_parallel(
    input_list,
    function,
    chunks=100,
    max_cpu=16,
    output_func=None,
    task_name="",
    **kwargs,
):
    """chunked_parallel.

    Args:
        input_list : list of objects to apply function
        function : Callable with 1 input and returning a single value
        chunks: number of chunks
        max_cpu: Max num cpus
        output_func: an output function that writes function output to the disk
    """
    # Adding it here fixes somessetting disrupted elsewhere

    def batch_func(list_inputs):
        outputs = []
        for i in list_inputs:
            outputs.append(function(i))
        return outputs

    list_len = len(input_list)
    if list_len == 0:
        raise ValueError('Empty list to process!')
    num_chunks = min(list_len, chunks)
    step_size = len(input_list) // num_chunks

    chunked_list = [
        input_list[i : i + step_size] for i in range(0, len(input_list), step_size)
    ]

    from pathos import multiprocessing as mp
    cpus = min(mp.cpu_count(), max_cpu)
    with mp.ProcessPool(processes=cpus, **kwargs) as pool:
        iter_outputs = tqdm(pool.imap(batch_func, chunked_list), total=len(chunked_list), desc=task_name)
        if output_func is None:
            list_outputs = list(iter_outputs)
            # Unroll
            full_output = [j for i in list_outputs for j in i]
            return full_output
        else:
            output_func(itertools.chain.from_iterable(iter_outputs))


def subprocess_parallel(cmd_list, max_parallel=4, max_parallel_per_gpu=None, gpus=None, env_list=None, delay_start=5):
    """
    Run command line jobs in parallel

    Args:
        cmd_list:
        max_parallel:
        max_parallel_per_gpu:
        gpus:
        env_list:

    Returns:

    """
    import subprocess
    from multiprocessing.dummy import Pool
    from multiprocessing import Manager
    import os

    manager = Manager()

    def terminate_processes():
        print("Terminating the pool...")
        try:
            pool.close()
            pool.terminate()
            pool.join()
        except Exception as e:
            print(f"Error terminating pool: {e}")

    def signal_handler(sig, frame):
        terminate_processes()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if env_list is None:
        env_list = [{} for _ in cmd_list]

    for idx, _ in enumerate(cmd_list):
        env_list[idx] = {
            "TQDM_POSITION": f'{idx + 1}',
            "TQDM_DESC": f"cmd {idx}/{len(cmd_list)}",
        }

    try:
        if gpus is None:
            def run_command(cmd_env_id):
                cmd, new_env, job_id = cmd_env_id
                if delay_start > 0:
                    time.sleep(delay_start * job_id)
                print(cmd + "\n")
                env = os.environ.copy()
                for k, v in new_env.items():
                    env[k] = v
                process = subprocess.Popen(cmd, shell=True, env=env)
                process.wait()

            with Pool(max_parallel) as pool:
                pool.map(run_command, zip(cmd_list, env_list, range(len(cmd_list))))

        else:
            gpu_job_count = manager.dict()
            locker = manager.Lock()

            if max_parallel_per_gpu is None:
                max_parallel_per_gpu = max_parallel

            gpus = list(gpus)  # Ensure gpus is a list
            for gpu in gpus:
                gpu_job_count[gpu] = 0

            def run_command(cmd, new_env, job_id, gpu_id):
                if delay_start > 0:
                    time.sleep(delay_start * job_id)
                print(f"Running on GPU {gpu_id}: {cmd}\n")
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                for k, v in new_env.items():
                    env[k] = v
                process = subprocess.Popen(cmd, shell=True, env=env)
                process.wait()

            def wrapper(cmd_env_id):
                nonlocal gpu_job_count
                while True:
                    with locker:
                        for gpu in gpus:
                            if gpu_job_count[gpu] < max_parallel_per_gpu:
                                gpu_job_count[gpu] += 1
                                selected_gpu = gpu
                                break
                        else:
                            continue
                        break
                try:
                    cmd, new_env, job_id = cmd_env_id
                    run_command(cmd, new_env, job_id, selected_gpu)
                finally:
                    with locker:
                        gpu_job_count[selected_gpu] -= 1

            with Pool(len(gpus) * max_parallel_per_gpu) as pool:
                pool.map(wrapper, zip(cmd_list, env_list, range(len(cmd_list))))
    except Exception as e:
        terminate_processes()
        raise e


def parallel_producer_consumer(producer_func, consumer_func, output_length, output_func=None,
                               producer_workers=0, consumer_workers=16):
    """
    If only producer_func, consumer_func, output_length are given, it will return a list of output
    If output_func is also given, output_func will do the post processing (e.g. write to disk), and this function
      will not return anything.
    producer_workers=0 means that the producer will run in the main process.
    consumer_func is None or consumer_workers=0 means that there is no consumers

    The functions should have the following characteristics:
      * producer_func(inp_q: JoinableQueue) -> None
      * consumer_func(data) -> output values
      * output_func(out_q, output_length) -> None

    Args:
        producer_func: must a function that accepts the JoinableQueue inp_q and put data into inp_q
        consumer_func: must be a function that accepts the data put by producer or None
        output_func: an output function that consumes the output queue (e.g. write items to disk)
        output_length: length of output list
        producer_workers: number of producer workers. If 0, producer will run on main thread
        consumer_workers: number of consumer workers

    Returns:
        output_list
    """
    import multiprocess.context as ctx
    from multiprocess.queues import JoinableQueue
    context = ctx.SpawnContext()

    producer_workers = min(context.cpu_count(), producer_workers)
    consumer_workers = min(context.cpu_count(), consumer_workers)

    inp_q = context.JoinableQueue()
    msg_q = context.JoinableQueue()

    # Define the processes
    def producer(inp_q: JoinableQueue):
        producer_func(inp_q)

    def consumer(inp_q: JoinableQueue, out_q: JoinableQueue, msg_q: JoinableQueue):
        msg_q.put('ALIVE')  # tell main thread for successfully starting the process
        while True:
            try:
                data = inp_q.get(block=True, timeout=0.5)

                return_val = consumer_func(data)

                # Put result to output queue
                out_q.put(return_val)

                inp_q.task_done()
            except queue.Empty:
                pass

    if output_func is not None:
        def output_worker(out_q: JoinableQueue, output_length: int, msg_q: JoinableQueue):
            msg_q.put('ALIVE')  # tell main thread for successfully starting the process
            output_func(out_q, output_length)
    else:
        output_worker = None

    if producer_workers > 0:
        producers = [
            context.Process(target=producer, args=(inp_q,))
            for _ in range(producer_workers)
        ]
    else:
        producers = []

    if consumer_func and consumer_workers > 0:
        out_q = context.JoinableQueue()
        consumers = [
            context.Process(target=consumer, args=(inp_q, out_q, msg_q), daemon=True)
            for _ in range(consumer_workers)
        ]
    else:
        consumers = []
        out_q = inp_q  # no consumer, merge out_q and inp_q into one

    if output_worker:
        output_workers = [
            context.Process(target=output_worker, args=(out_q, output_length, msg_q), daemon=True)
        ]
    else:
        output_workers = []

    # Exception handlers
    def terminate_processes():
        for p in consumers + producers + output_workers:
            p.terminate()
            p.join()

    def signal_handler(sig, frame):
        terminate_processes()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # start the processes (NOTE: list could be empty)
        for p in consumers + producers + output_workers:
            p.start()

        # block before all consumers and output workers are alive
        alive_counter = 0
        while alive_counter < len(consumers) + len(output_workers):
            try:
                val = msg_q.get()
                if val == 'ALIVE':
                    alive_counter += 1
                msg_q.task_done()
            except queue.Empty:
                pass

        # wait for producer to finish
        if producer_workers == 0:
            producer(inp_q)  # call producer function in the main process
        else:
            for p in producers:
                p.join()  # wait for producer processes to finish

        # get output
        if not output_worker:
            # get output manually
            progress_bar = tqdm(total=output_length)
            output_list = []
            while len(output_list) < output_length:
                try:
                    output_list.append(out_q.get())
                    progress_bar.update()
                    out_q.task_done()
                except queue.Empty:
                    pass
            progress_bar.close()
            return output_list
        else:
            for p in output_workers:
                p.join()  # wait for output workers to finish
    except Exception as e:
        terminate_processes()
        raise e
