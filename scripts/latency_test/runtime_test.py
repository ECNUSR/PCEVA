import os
import torch
import time
import pathlib
import logging
import argparse
import numpy as np
import torch.nn.functional as F
import torch_tensorrt

from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

import models
import dataset as dd

from utils import util_logger
from utils import util_image as util
from utils.model_summary import get_model_flops


def main(args):

    """
    BASIC SETTINGS
    """
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   
    """
    LOAD MODEL
    """
    model = models.__dict__[args.model_name]()
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
            
    model = model.to(device)
    
    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))
          
            
    """
    SETUP RUNTIME
    """
    test_results = OrderedDict()
    test_results["runtime"] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    """
    TESTING
    """  
    input_data = torch.randn((1, 3, args.crop_size[0]//args.scale, args.crop_size[1]//args.scale)).to(device)
    
    if args.fp16:
        input_data = input_data.half()
        model = model.half()
    
    # GPU warmp up
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(input_data)
            
    print("Start timing ...")
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in tqdm(range(args.repeat)):       
            start.record()
            _ = model(input_data)
            end.record()

            torch.cuda.synchronize()
              
            test_results["runtime"].append(start.elapsed_time(end))  # milliseconds

        ave_runtime = sum(test_results["runtime"]) / len(test_results["runtime"])
        logger.info('------> Average runtime of ({}) is : {:.6f} ms'.format(args.submission_id, ave_runtime / args.batch_size))
        

        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # specify submission
    parser.add_argument("--model-name", type=str)

    # specify dirs
    parser.add_argument("--save-dir", type=str, default="internal")
    
    # specify test case
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=244)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, nargs="+", default=[3840, 2160])  
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()
        
    main(args)