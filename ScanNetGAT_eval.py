import argparse
import numpy as np
from tqdm import tqdm
import torch

from ScanNetGQA_dataset import ScanNetGQA
from pipeline_model_gat import PipelineModel

def parse_args():
    parser = argparse.ArgumentParser('Explainable GQA Parser', add_help=False)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=32)
    return parser.parse_args()

def main():
    args = parse_args()
    cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ScanNetGQA dataset
    test_dataset = ScanNetGQA() # for test need to build vocab
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = args.batch_size,
        sampler = test_sampler,
        drop_last=False,
        collate_fn = None, #TODO
        num_workers=args.workers
    )

    # Load model
    model = PipelineModel()