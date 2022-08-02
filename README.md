# Wandb Parallelization

## Step 1: sweep_seed_parallel.yaml

Please see the weep_seed_parallel.yaml file for an example.

## Step 2: run_wandb_sweep.sh 

Please see the run_wandb_sweep.sh file for an example.
For Sbatch: set array to how many sets of hyperparameters you would like to run in parallel (e.g. #SBATCH --array=1-80 runs 80 sets of hyperparameters in parallel)
Note that time can be set to something very small (e.g. 10 minutes) since the python script will simply call another Bash file and should terminate quickly.

Make sure that your Bash script contains the following line:

    wandb agent --count 1 <USERNAME/PROJECTNAME/SWEEPID>

## Step 3: wandb_com.py

Please see the wandb_com.py file for an example.
Make sure that the argument parser in this python file takes inputs that match the variables in the Yaml file.

The following code snippet must be present in wandb_com.py so that runs will be grouped by hyperparameter/sweep:

    sweep_run = wandb.init(settings=wandb.Settings(start_method="fork")) # Set starting method to avoid well-documented error: https://docs.wandb.ai/guides/track/launch#init-start-error
	sweep_id = sweep_run.sweep_id or "unknown"
	project_url = sweep_run.get_project_url()
	sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
	sweep_run.notes = sweep_group_url
	sweep_run.save()
	sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

This python script will then launch another Bash file (see step 4).

## Step 4: wandb_run_train.sh 

Please see the wandb_run_train.sh file for an example.
For Sbatch: set array to how many seeds you would like to test with the current set of hyperparameters (e.g. #SBATCH --array=1-3 will run 3 folds/seeds for the current set of hyperparameters).

Feel free to take out the 
    export LD_DEBUG=files,libs 
and 
    unset LD_DEBUG
lines. They are simply there because sometimes the error "/usr/bin/lua: error while loading shared libraries: libtinfo.so.6: cannot open shared object file: No such file or directory" pops up on some compute nodes. It seems like some other users have encountered the same problem, so IT suggested that I add these two lines in to help them debug in the future. These two lines will print a bunch of garbage to the err outfile, but as long as you see the 

    [=== Module openmpi/4.0.4 loaded ===]
    [=== Module pytorch/1.7.0 loaded ===]

lines in the error outfile everything is fine.

This bash file will then open your main python training file.

## Step 5: train_wandb.py

Make sure you include the following code snippet before you log anything with Wandb (see train_wandb.py for example):

    reset_wandb_env()
    run_name = "{}-{}".format(sweep_run_name, fold)
    run = wandb.init(
        group=sweep_id,
        job_type=sweep_run_name,
        name=run_name,
        config=argsdict
    )
    run = wandb.init(settings=wandb.Settings(start_method="fork"))

Note that the reset_wandb_env() function (see example train_wandb.py file) must be called. Otherwise, runs will override each other. 

The following tutorial explains how to log runs with Wandb: https://docs.wandb.ai/guides/sweeps/quickstart. Note: since we initialized wandb with the name run, we must log stats with run.log(...) rather than wandb.log(...).

## Misc.

Once all of the files above have been configured, running everything in parallel takes two steps:

1. Once the Yaml file has been created, use the 

    wandb sweep sweep_seed_parallel.yaml

command to generate a wandb sweep agent. An example of the accompanying Wandb sweep_id would be e5v18wr8. You will need this sweep_id for the next step.

2. Using the sweep_id generated above, call run_wandb_sweep.sh using:

    sbatch run_wandb_sweep.sh sweep_id

, where sweep_id is the sweep_id generated in step 1.

That's it!