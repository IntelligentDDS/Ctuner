# Ctuner
CTuner: Automatic  Distributed NoSQL Database Tuning with Causal Reinforcement Learning

## Summary

Configuration tuning is always an effective way to improve the performance of NoSQL DB. Recently, reinforcement learning (RL) has shown great potential in the performance tuning of databases. However, using RL for NoSQL database tuning is still challenging because NoSQL databases usually have a large number of tunable knobs.Moreover, compared with the heuristic algorithm, RL has a cold start problem in the early stage of offline training, which leads to an increase in time cost. On the other hand, it is difficult for RL models to adapt quickly to unseen environments. To address these issues, we propose a NoSQL tuning system named CTuner, which recommends configurations efficiently and effectively. In order to reduce the total cost of tuning, 1) CTuner uses Bayesian Optimization to generate high-quality training samples to solve the
cold start problem, and Random Forest is used to select important knobs; 2) CTuner uses causal inference to improve the exploitation strategy of TD3. Meanwhile, a novel Multi-tasking Prioritized Replay Memory which fuses reward and Temporal Difference error is designed; 3) Meta-learning is introduced to improve the adaptability of the tuning model. Our experiments with YCSB benchmark show that CTuner can find a better configuration with the same
time cost, with throughput increased by 2.4%-27.4% and 95%-tail latency decreased by 1.2%-13.2%.


## Development environment

| hostname   | ip             | required      | Function               | user | password   | operating system    | Python version |
| -------- | -------------- | ------------- | ------------------ | -------- | ------ | ----------- | ---------- |
| CTuner1 | 33.33.33.104 | openjdk-8-jdk | server  | ubuntu      | ubuntu | Ubuntu18.04 | 3.7        |
| CTuner2 | 33.33.33.179 | openjdk-8-jdk | client        | ubuntu      | ubuntu | Ubuntu18.04 | 3.7        |


## Step1: Install basic software for server and client 
Install openjdk-8-jdk on two hosts
```shell
mgt@CTuner1: sudo apt update && sudo apt -y upgrade
mgt@CTuner1: sudo apt-get install openjdk-8-jdk
mgt@CTuner1: java -version

mgt@CTuner2: sudo apt update && sudo apt -y upgrade
mgt@CTuner2: sudo apt-get install openjdk-8-jdk
mgt@CTuner2: java -version
```

##Step2: Install python3.7 for Server
Install environment dependencies
```shell
mgt@CTuner1: sudo apt-get install zlib1g-dev libbz2-dev libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev tk-dev libgdbm-dev libdb-dev libpcap-dev xz-utils libexpat1-dev liblzma-dev libffi-dev libc6-dev
```
Create an installation directory
```shell
mgt@CTuner1: sudo mkdir -p /usr/local/python3.7 
```
Download python3.7 and unzip the download installation package
```shell
mgt@CTuner1: wget https://www.python.org/ftp/python/3.7.8/Python-3.7.8.tgz
mgt@CTuner1: tar -zxvf Python-3.7.8.tgz
```
Enter the unzipped directory
```shell
mgt@CTuner1: cd Python-3.7.8/
```
Compile and install
--prefix: used to specify the installation location
--enable-optimizations: for optimization configuration
```shell
mgt@CTuner1: ./configure --prefix=/usr/local/python3.7 --enable-optimizations
```
```shell
mgt@CTuner1: make
```
In order to prevent replacing the default installed version of the system, use altinstall
```shell
mgt@CTuner1: sudo make altinstall
```
Create soft links to python3.7 and pip3.7
```shell
mgt@CTuner1: sudo ln -sf /usr/local/python3.7/bin/python3.7 /usr/bin/python3
mgt@CTuner1: sudo ln -sf /usr/local/python3.7/bin/pip3.7 /usr/bin/pip3
```
Test whether the installation is successful
```shell
mgt@CTuner1: python3
```

## Step3: Install an automated operation and maintenance tool called ansible
Install ansible for server
```shell
mgt@CTuner1: pip3 install  ansible==2.9.27 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
mgt@CTuner1: sudo apt install ansible
mgt@CTuner1: ansible --version
```
Install python for client
```shell
mgt@CTuner2: sudo apt-get install python
```
Ansible of the server connects to the client through SSH, first generates a public key key on the server node, and then copies it to the client node.
```shell
mgt@CTuner1: ssh-keygen 
mgt@CTuner1: ls -l /home/ubuntu/.ssh
```
The server copies the public key key to the client node:
```shell
mgt@CTuner1: ssh-copy-id ubuntu@33.33.33.179
mgt@CTuner1: ssh-copy-id ubuntu@33.33.33.104
```
Test whether it is connected, it is successful without entering a password
```shell
mgt@CTuner1: ssh ubuntu@33.33.33.179      
```
Edit /etc/ansible/hosts to add client information
```shell
mgt@CTuner1: sudo vim /etc/ansible/hosts
```
Add content as follows
```
[nosql]
nosql-tester ansible_ssh_host=33.33.33.104 ansible_ssh_user=ubuntu
nosql-testee ansible_ssh_host=33.33.33.179 ansible_ssh_user=ubuntu
```
Test that the ping command is executed successfully
```shell
mgt@CTuner1: ansible nosql -m ping
```

## Step4: Install python dependent environment
Install related dependencies
```shell
mgt@CTuner1: sudo apt-get install graphviz graphviz-dev
mgt@CTuner1: pip3 install -r requirements.txt  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

## Step5: Modify the config
Modify the config file information under the environment
```python
  'mongodb': {
        'tester': 'mongo-tester',   # Need to be modified according to the actual situation
        'testee': 'mongo-testee',   # Need to be modified according to the actual situation
        'host': '33.33.33.179',   # Need to be modified according to the actual situation
        'user': 'ubuntu',             # Need to be modified according to the actual situation
        'password': 'ubuntu',     # Need to be modified according to the actual situation
        'port': 27017,
        'database': 'ycsb',               # fixed
        'memory': 34359738368,
        'test_mode': 'synchronization'    # fixed
  },
```

## Step6: High Quality Sample Acquisition command

Take mongodb to obtain high-quality samples as an example, meta_train_01 means calling the configuration file in environment\target\mongodb\tests\meta_train_01.yml
```shell
mgt@CTuner1: cd tuner
mgt@CTuner1: python3 meta_preheat.py mongodb meta_train_01
```

## Step7: Train command

Taking mongodb meta training as an example, meta_train_01 means calling the configuration file in environment\target\mongodb\tests\meta_train_01.yml
```shell
mgt@CTuner1: cd tuner
mgt@CTuner1: python3 meta_train.py mongodb meta_train_01
```

## Step8: test command

Taking mongodb evaluating as an example, meta_eval_01 means calling the configuration file in environment\target\mongodb\tests\meta_eval_01.yml
```shell
mgt@CTuner1: cd tuner
mgt@CTuner1: python3 meta_evaluate.py mongodb meta_eval_01
```

## Possible problems:
1、
subprocess.CalledProcessError: Command '('lsb_release', '-a')' returned non-zero exit status 1.

Solution:

```shell
mgt@CTuner1: sudo find / -name 'lsb_release.py'
mgt@CTuner1: sudo cp  /usr/lib/python3/dist-packages/lsb_release.py /usr/local/python3.7/lib/python3.7/
```

