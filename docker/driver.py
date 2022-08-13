# Submit job to the remote cluster

import datetime
import os
import pickle
import random
import subprocess
import sys
import time
import json

import yaml


def flatten(d):
    out = {}
    for key, val in d.items():
        if isinstance(val, dict):
            val = [val]
        if isinstance(val, list):
            for subdict in val:
                deeper = flatten(subdict).items()
                out.update({key2: val2 for key2, val2 in deeper})
        else:
            out[key] = val
    return out


def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data


def process_cmd(json_file, local=False):
    json_conf = load_json_conf(json_file)

    # if json_conf["dataset"] == "femnist" or json_conf["dataset"] == "reddit":
    #     process_cmd_json(json_file, local = local)
    #     return None
    
    os.environ["CUDA_VISIBLE_DEVICES"]="2"

    yaml_conf = load_yaml_conf("./base_conf.yml")

    ps_ip = yaml_conf['ps_ip']
    worker_ips, total_gpus = [], []
    cmd_script_list = []

    executor_configs = "=".join(yaml_conf['worker_ips'])
    for ip_gpu in yaml_conf['worker_ips']:
        ip, gpu_list = ip_gpu.strip().split(':')
        worker_ips.append(ip)
        total_gpus.append(eval(gpu_list))

    time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%m%d_%H%M%S')
    running_vms = set()
    job_name = 'fedscale_job'
    log_path = './logs'
    submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(yaml_conf['auth']['ssh_user']) else ""

    job_conf = {'time_stamp': time_stamp,
                'ps_ip': ps_ip,
                }

    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)

    try:
        ps_port = yaml_conf['ps_port']
        conf_script = f' --ps_port={ps_port}'
    except:
        conf_script = ''
    setup_cmd = ''
    if yaml_conf['setup_commands'] is not None:
        setup_cmd += (yaml_conf['setup_commands'][0] + ' && ')
        for item in yaml_conf['setup_commands'][1:]:
            setup_cmd += (item + ' && ')

    cmd_sufix = f" "

    for conf_name in job_conf:
        if conf_name == "job_name":
            job_conf[conf_name] = json_conf["dataset"] + '+' + json_conf["model"]
        elif conf_name == "task":
            if json_conf['dataset'] != 'femnist':
                job_conf[conf_name] = 'cv'
            else:
                job_conf[conf_name] = "simple" # TO-DO ?
        elif conf_name == "num_participants":
            job_conf[conf_name] = json_conf["training_param"]["client_per_round"]
        elif conf_name == "data_set":
            if json_conf['dataset'] == 'femnist':
                job_conf[conf_name] = 'femnist2'
            else:
                job_conf[conf_name] = json_conf["dataset"]
        elif conf_name == "data_dir":
            if json_conf['dataset'] == 'femnist':
                job_conf[conf_name] = "../data/" + json_conf["dataset"]
            else:
                job_conf[conf_name] = "../data/csv_data/" + json_conf["dataset"]
        elif conf_name == "model":
            job_conf[conf_name] = json_conf["model"]
        elif conf_name == "gradient_policy":
            job_conf[conf_name] = json_conf["algorithm"]
        elif conf_name == "eval_interval":
            job_conf[conf_name] = json_conf["training_param"]["epochs"] + 1
        elif conf_name == "rounds":
            job_conf[conf_name] = json_conf["training_param"]["epochs"] + 2
        elif conf_name == "inner_step":
            job_conf[conf_name] = json_conf["training_param"]["inner_step"]
        elif conf_name == "learning_rate":
            job_conf[conf_name] = json_conf["training_param"]["learning_rate"]
        elif conf_name == "batch_size":
            job_conf[conf_name] = json_conf["training_param"]["batch_size"]
        elif conf_name == "use_cuda":
            job_conf[conf_name] = (json_conf["bench_param"]["device"] == "gpu")

        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'
        if conf_name == "job_name":
            job_name = job_conf[conf_name]
        if conf_name == "log_path":
            log_path = os.path.join(
                job_conf[conf_name], 'log', job_name, time_stamp)

    if json_conf['dataset'] == 'femnist':
        # job_conf['data_set'] = 'femnist2'
        # job_conf['temp_tag'] = 'simple_femnist'
        conf_script = conf_script + ' --temp_tag=simple_femnist'

    print(conf_script)

    total_gpu_processes = sum([sum(x) for x in total_gpus])
    # =========== Submit job to parameter server ============
    running_vms.add(ps_ip)
    ps_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['aggregator_entry']} {conf_script} --this_rank=0 --num_executors={total_gpu_processes} --executor_configs={executor_configs} "

    with open(f"{job_name}_logging", 'wb') as fout:
        pass

    print(f"Starting aggregator on {ps_ip}...")
    with open(f"{job_name}_logging", 'a') as fout:
        if local:
            subprocess.Popen(f'{ps_cmd}', shell=True, stdout=fout, stderr=fout)
        else:
            subprocess.Popen(f'ssh {submit_user}{ps_ip} "{setup_cmd} {ps_cmd}"',
                             shell=True, stdout=fout, stderr=fout)

    time.sleep(10)
    # =========== Submit job to each worker ============
    rank_id = 1
    
    for worker, gpu in zip(worker_ips, total_gpus):
        running_vms.add(worker)
        print(f"Starting workers on {worker} ...")

        for cuda_id in range(len(gpu)):
            for _ in range(gpu[cuda_id]):
                worker_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['executor_entry']} {conf_script} --this_rank={rank_id} --num_executors={total_gpu_processes} --cuda_device=cuda:{cuda_id} "
                rank_id += 1

                with open(f"{job_name}_logging", 'a') as fout:
                    time.sleep(5)
                    if local:
                        subprocess.Popen(f'{worker_cmd}',
                                         shell=True, stdout=fout, stderr=fout)
                    else:
                        subprocess.Popen(f'ssh {submit_user}{worker} "{setup_cmd} {worker_cmd}"',
                                         shell=True, stdout=fout, stderr=fout)

    # dump the address of running workers
    current_path = os.path.dirname(os.path.abspath(__file__))
    job_name = os.path.join(current_path, job_name)
    with open(job_name, 'wb') as fout:
        job_meta = {'user': submit_user, 'vms': running_vms}
        pickle.dump(job_meta, fout)

    print(f"Submitted job, please check your logs {job_conf['log_path']}/logs/{job_conf['job_name']}/{time_stamp} for status")


def load_json_conf(json_file):
    with open(json_file) as fin:
        data = json.load(fin)
    return data


def process_cmd_json(json_file, local=False):
    json_conf = load_json_conf(json_file)

    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    
    bench_para = json_conf['bench_param']
    if bench_para['mode'] == 'local':
        ps_ip = 'localhost'
        local = True
    else:
        local = False
    
    worker_ips, total_gpus = [], []
    cmd_script_list = []

    if local:
        executor_configs = 'localhost:[2]'
        lis_worker_ips = ['localhost:[2]']    
        for ip_gpu in lis_worker_ips:
            ip, gpu_list = ip_gpu.strip().split(':')
            worker_ips.append(ip)
            total_gpus.append(eval(gpu_list))
    else:
        lis_worker_ips = bench_para['hosts'][1:]
        ps_ip = bench_para['hosts'][0]['hostname']
        executor_configs = ''
        gpu_cnt = 0
        for tmp_gpu in lis_worker_ips:
            worker_ips.append(tmp_gpu['hostname'])
            total_gpus.append(eval([2]))
            if gpu_cnt == 0:
                executor_configs = tmp_gpu['hostname'] + ':[2]'
            else:
                executor_configs = executor_configs + '=' + tmp_gpu['hostname'] + ':[2]'
            gpu_cnt = gpu_cnt + 1

    time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%m%d_%H%M%S')
    running_vms = set()
    job_name = 'fedscale_job'
    log_path = './logs'

    submit_user = ""
    
    job_conf = {'time_stamp': time_stamp,
                'ps_ip': ps_ip,
                'job_name': json_conf['framework'],
                }
    
    if job_conf['job_name'] == 'femnist':
        job_conf['tmp_tag'] = 'simple_femnist'

    
    for conf_name,conf in json_conf['training_param'].items():
        if conf_name != "optimizer_param" and conf_name != "optimizer_param" and conf_name != 'tree_param':
            if conf_name == 'epochs':
                conf_name = 'rounds'
            if conf_name == 'client_per_round':
                conf_name = 'num_participants'
            
            job_conf[conf_name] = conf

    try:
        ps_port = job_conf['ps_port']
        conf_script = f' --ps_port={ps_port}'
    except: 
        conf_script = ''

    setup_cmd = 'source $HOME/anaconda3/bin/activate fedscale && '

    cmd_sufix = f" "

    for conf_name in job_conf:
        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'
        if conf_name == "job_name":
            job_name = job_conf[conf_name]
        if conf_name == "log_path":
            log_path = os.path.join(
                job_conf[conf_name], 'log', job_name, time_stamp)

    total_gpu_processes = sum([sum(x) for x in total_gpus])
    # =========== Submit job to parameter server ============
    running_vms.add(ps_ip)
    ps_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['aggregator_entry']} {conf_script} --this_rank=0 --num_executors={total_gpu_processes} --executor_configs={executor_configs} "

    with open(f"{job_name}_logging", 'wb') as fout:
        pass

    print(f"Starting aggregator on {ps_ip}...")
    with open(f"{job_name}_logging", 'a') as fout:
        if local:
            subprocess.Popen(f'{ps_cmd}', shell=True, stdout=fout, stderr=fout)
        else:
            subprocess.Popen(f'ssh {submit_user}{ps_ip} "{setup_cmd} {ps_cmd}"',
                             shell=True, stdout=fout, stderr=fout)

    time.sleep(10)
    # =========== Submit job to each worker ============
    rank_id = 1
    
    for worker, gpu in zip(worker_ips, total_gpus):
        running_vms.add(worker)
        print(f"Starting workers on {worker} ...")

        for cuda_id in range(len(gpu)):
            for _ in range(gpu[cuda_id]):
                worker_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['executor_entry']} {conf_script} --this_rank={rank_id} --num_executors={total_gpu_processes} --cuda_device=cuda:{cuda_id} "
                rank_id += 1

                with open(f"{job_name}_logging", 'a') as fout:
                    time.sleep(5)
                    if local:
                        subprocess.Popen(f'{worker_cmd}',
                                         shell=True, stdout=fout, stderr=fout)
                    else:
                        subprocess.Popen(f'ssh {submit_user}{worker} "{setup_cmd} {worker_cmd}"',
                                         shell=True, stdout=fout, stderr=fout)

    # dump the address of running workers
    current_path = os.path.dirname(os.path.abspath(__file__))
    job_name = os.path.join(current_path, job_name)
    with open(job_name, 'wb') as fout:
        job_meta = {'user': submit_user, 'vms': running_vms}
        pickle.dump(job_meta, fout)

    print(f"Submitted job, please check your logs {job_conf['log_path']}/logs/{job_conf['job_name']}/{time_stamp} for status")



def terminate(job_name):

    current_path = os.path.dirname(os.path.abspath(__file__))
    job_meta_path = os.path.join(current_path, job_name)

    if not os.path.isfile(job_meta_path):
        print(f"Fail to terminate {job_name}, as it does not exist")

    with open(job_meta_path, 'rb') as fin:
        job_meta = pickle.load(fin)

    for vm_ip in job_meta['vms']:
        print(f"Shutting down job on {vm_ip}")
        with open(f"{job_name}_logging", 'a') as fout:
            subprocess.Popen(f'ssh {job_meta["user"]}{vm_ip} "python {current_path}/shutdown.py {job_name}"',
                             shell=True, stdout=fout, stderr=fout)

print_help: bool = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'submit' or sys.argv[1] == 'start':
        process_cmd(sys.argv[2], False if sys.argv[1] == 'submit' else True)
        # process_cmd_json(sys.argv[2], False if sys.argv[1] == 'submit' else True)
    elif sys.argv[1] == 'stop':
        terminate(sys.argv[2])
    else:
        print_help = True
else:
    print_help = True

if print_help:
    print("\033[0;32mUsage:\033[0;0m\n")
    print("submit $PATH_TO_CONF_YML     # Submit a job")
    print("stop $JOB_NAME               # Terminate a job")
    print()
