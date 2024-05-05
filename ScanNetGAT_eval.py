import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import json

from ScanNetGQA_dataset import ScanNetGQA, ScanNetGQA_collate_fn
from pipeline_model_gat import PipelineModel

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser('Explainable GQA Parser', add_help=False)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=32)
    # parser.add_argument('--lr_drop', default=30, type=int)
    # parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', dest='lr')
    parser.add_argument('--ckpt', default='./outputdir/checkpoint.pth', type=str, metavar='PATH', help='path to latest checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load ScanNetGQA dataset
    test_dataset = ScanNetGQA(split='test') # for test need to build vocab?
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = args.batch_size,
        sampler = test_sampler,
        drop_last = False,
        collate_fn = ScanNetGQA_collate_fn,
        num_workers = args.workers,
        shuffle = False
    )

    # Load model
    model = PipelineModel()
    model = model.to(device=cuda)

    # Load latest model checkpoint    
    ckpt = torch.load(args.ckpt, map_location=torch.device(cuda))
    model.load_state_dict(ckpt['model'])

    # Run inference
    test(test_loader, model, DUMP_RESULT=False)
    return

def test(loader, model, DUMP_RESULT=False):
    model.eval()
    with torch.no_grad():
        for i, (data_batch) in enumerate(tqdm(loader)):
            question_id, question_text, question_tokenzied, sg, answer, answer_label = data_batch

            question_tokenzied = question_tokenzied.to(device=cuda, non_blocking=True)
            sg = sg.to(device=cuda, non_blocking=True)
            
            programs_output, short_answer_logits = model(
                question_tokenzied,
                sg,
                None,
                None,
                SAMPLE_FLAG=True)
            
            short_answer_logits = F.softmax(short_answer_logits, dim=1)

            # Convert to numpy
            # programs_output = programs_output.detach().cpu().numpy()
            short_answer_logits = short_answer_logits.detach().cpu().numpy()
            
            answer_id = np.argmax(short_answer_logits, axis=1)
            with open('./meta_info/trainval_label2ans.json') as infile:
                label2ans = json.load(infile)

            for i in range(len(answer_id)):
                print('Question: ', question_text[i])
                print('GT Answer: ', answer_label[i][0])
                print('Predicted answer: ', label2ans[answer_id[i]])
                print('Probability', short_answer_logits[i, answer_id[i]])
                # print('Predicted program: ', ScanNetGQA.indices_to_string(programs_output[i], True))

            # print(programs_output.shape)
            # print(short_answer_logits.shape)


if __name__ == '__main__':
    main()