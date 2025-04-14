#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u train.py --name 'irra_added'
--loss_names 'sdm+id+mlm+mcq'
--dataset_name "CUHK-PEDES"
--mcq_temperature 0.04

