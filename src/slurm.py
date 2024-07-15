from argparse import ArgumentParser
from submitit.helpers import DelayedSubmission
from submitit.helpers import TorchDistributedEnvironment


def add_submitit_args(parser: ArgumentParser): 
    # check if "exp_dir" is already present in the parser
    add_exp_dir = True
    for action in parser._actions: 
        if action.dest == "exp_dir": 
            add_exp_dir = False
            break 
    if add_exp_dir:
        parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory.")
    parser.add_argument("--time", type=int, default=60, help="Time in minutes.")
    parser.add_argument("--partition", type=str, default="a40,t4v2,rtx6000", help="Partition to submit job to.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--cpus_per_task", type=int, default=10, help="Number of CPUs to use.")
    parser.add_argument("--mem", type=str, default='16G', help="Memory.")


class SubmititRunner: 
    def __init__(self, target, args): 
        self.target = target
        self.args = args

    def __call__(self): 
        TorchDistributedEnvironment().export()
        self.target(self.args)

    def checkpoint(self): 
        return DelayedSubmission(
            SubmititRunner(self.target, self.args)  
        )


def submit_job(target, args, setup=[]): 
    import submitit
    executor = submitit.SlurmExecutor(folder=args.exp_dir, max_num_timeout=10)
    executor.update_parameters(
        time=args.time,
        partition=args.partition,
        gres=f"gpu:{args.gpus}",
        cpus_per_task=args.cpus_per_task,
        mem=args.mem, 
        setup=setup, 
        ntasks_per_node=args.gpus, 
    )

    job = executor.submit(SubmititRunner(target, args))
    print(f"Submitted job {job.job_id}")
    print(f"Logs at {job.paths.stdout}")