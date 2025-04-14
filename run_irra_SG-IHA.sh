#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u train.py --name 'dual_encoder_added'
--loss_names 'sdm+id'
--dataset_name "CUHK-PEDES"
--mcq_temperature 0.04

