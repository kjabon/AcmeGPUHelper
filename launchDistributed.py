# The point of this file is to show, in Acme/Launchpad, how to set per-process environment variables for gpu memory allocation.

# E.g., the following block will do just that, by passing in a dictionary of env variables to the Launchpad "PythonProcess" class.
# Launchpad will then take this dictionary describing all resources available to each process type, and set about launching each with lp.launch.
# 'learner':
# PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1',
#                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform',
#                                   XLA_PYTHON_CLIENT_MEM_FRACTION='.40', TF_FORCE_GPU_ALLOW_GROWTH='true'))

def launchDistributed(experiment_config, numActors=1, numLearners=1):
    print("______________________________________________")
    print("numactors: {}, numlearners: {}".format(numActors, numLearners))
    print("______________________________________________")
    time.sleep(2)
    program = experiments.make_distributed_experiment(
        experiment=experiment_config,
        num_actors=numActors, num_learner_nodes=numLearners)
    resources = {
        # The 'actor' and 'evaluator' keys refer to
        # the Launchpad resource groups created by Program.group()
        'actor':
            PythonProcess(  # Dataclass used to specify env vars and args (flags) passed to the Python process
                env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                         XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'evaluator':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'counter':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'replay':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'learner':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform',
                                   XLA_PYTHON_CLIENT_MEM_FRACTION='.40', TF_FORCE_GPU_ALLOW_GROWTH='true')),
    }

    worker = lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program),
                       launch_type=launchpad.context.LaunchType.LOCAL_MULTI_PROCESSING,
                       terminal=terminals[1], local_resources=resources)
    worker.wait()
