# -*- coding: utf-8 -*-

"""
description: MongoDB Database Configurations
"""

PROJECT_DIR = "/root/CDLTune"    # modify project path
instance_config = {
    'mongodb': {
        'tester': 'mongo-tester',   # Need to be modified according to the actual situation
        'testee': 'mongo-testee',   # Need to be modified according to the actual situation
        'host': '10.186.133.122',   # Need to be modified according to the actual situation
        'user': 'root',             # Need to be modified according to the actual situation
        'password': 'Chen0031',     # Need to be modified according to the actual situation
        'port': 27017,
        'database': 'ycsb',               # fixed
        'memory': 34359738368,
        'test_mode': 'synchronization'    # fixed
    },
    'cassandra': {
        'tester': 'mongo-tester',   # Need to be modified according to the actual situation
        'testee': 'mongo-testee',   # Need to be modified according to the actual situation
        'host': '10.186.133.122',   # Need to be modified according to the actual situation
        'user': 'root',             # Need to be modified according to the actual situation
        'password': 'Chen0031',     # Need to be modified according to the actual situation
        'port': 27017,
        'database': 'ycsb',              # fixed
        'memory': 34359738368,
        'test_mode': 'asynchronization'  # fixed
    },
    'redis':{
        'tester': 'mongo-tester',   # Need to be modified according to the actual situation
        'testee': 'mongo-testee',   # Need to be modified according to the actual situation
        'host': '10.186.133.122',   # Need to be modified according to the actual situation
        'user': 'root',             # Need to be modified according to the actual situation
        'password': 'Chen0031',     # Need to be modified according to the actual situation
        'port': 27017,
        'database': 'ycsb',              # fixed
        'memory': 34359738368,
        'test_mode': 'asynchronization'  # fixed
    }
}