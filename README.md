Check out the companion [blog post](kjabon.github.io/blog/2023/AcmeIssues/).

# AcmeGPUHelper
Acme, Deepmind's RL lib, does not parallelize using multiprocessing out of the box on custom setups. This repo has fixes for the issues I ran into. The crux of the problem lies in memory allocation. Depending on your training code, you will possibly use both JAX and Tensorflow, both of which will independently try to allocate memory for themselves.

# Multiple processes
Please note that environment variables need to be set for all spawned processes. Launchpad, used by Acme, does not copy the current environment variables to new processes by default. See launchDistributed.py for more on this.

# Solving JAX annoyances
In the interest of being able to use our GPUs for as many processes as possible, we want to avoid allocating more memory than is needed per process.
To do this, the key environmental variables we need to set are XLA_PYTHON_CLIENT_ALLOCATOR (set to "platform" to allow Python to allocate memory as needed) and XLA_PYTHON_CLIENT_PREALLOCATE set to "false". See gpu.py for more.

Additionally, JAX cannot handle running pmapped functions on two or more GPUs if those GPUs are not identical. Frustrating, but true for the time being. Thus, for each process, we must allow only one set of identical GPUs to be visible at a time; this cannot be changed once you've started making models and tensors, so you have to set it once at the beginning. If you have an RTX 3060 and an RTX 3080, you set the environment variable CUDA_VISIBLE_DEVICES to "0" OR "1". If you have two 3060s and one 3080, you set CUDA_VISIBLE_DEVICES to "0,1" OR "2". To avoid confusion while using nvidia-smi in the terminal to monitor my GPU utilization, I set CUDA_DEVICE_ORDER to "PCI_BUS_ID" to get the same ordering.

If the particular GPU doesn't matter, you're trying to spawn multiple training runs simultaneously from a bash script, and you're fairly confident you don't run the risk of overallocating, I like to use the GPUtil python library to pick the least utilized GPU. See gpu.py for more.

# Solving Tensorflow annoyances
For efficiency, Tensorflow by default allocates most of the memory of all GPUs.
This is problematic if you would like to use GPUs from multiple processes simultaneously. If you're training RL agents in parallel, this use case is practically a given, whether you are using multiple actors/learners or multiple training runs. Any process which interacts with Tensorflow (even if no Tensors are used!), in my experience, will automatically allocate all memory for the visible GPUs. 

To solve this, we must set memory growth for Tensorflow to True. See gpu.py for the code that does this. The tensorflow library function that accomplishes this task is equivalent to setting the TF_FORCE_GPU_ALLOW_GROWTH env variable directly, which needs to be done if you're spawning processes with Launchpad.



