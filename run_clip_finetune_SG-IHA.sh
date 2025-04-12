#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u train.py --name 'dual_encoder_sdm_id_seed42'
--loss_names 'sdm+id'
--dataset_name "CUHK-PEDES"
--mcq_temperature 0.04

