import os
import json
import torchtext

from ScanNetGQA_utils import load_config
from gqa_dataset_entry import GQATorchDataset


config = load_config()
dataset_path = config['path']['dataset_path']
datasets = {
    'train': str(),
    'val': str(),
    'test': str(os.path.join(dataset_path, 'ScanQA_v1.0_val.json'))
}

class ScanNetGQA():
    TEXT = torchtext.data.Field(
        sequential=True,
        tokenize="spacy",
        init_token="<start>",
        eos_token="<end>",
        tokenizer_language='en_core_web_sm',
        include_lengths=False,
        batch_first=False # Whether to produce tensors with the batch dimension first.
    )
    
    def __init__(self, split):
        self.split = split
        # assert split in
        dataset_path = datasets[split]
        with open(dataset_path) as infile:
            self.questions_data = json.load(infile)

        # TODO: load/build vocab
        # TODO: load graph dataset

        print(f'finished loading the data, totally {len(self.data)} instances')

    def __getitem__(self, index):
        # Load data
        item_q = self.questions_data[index]
        scene_id = item_q['scene_id']
        question_text = item_q['question']
        question_id = item_q['question_id']
        text_answers = item_q['answers']
        instances_answers = item_q['object_ids']
        objects_answers = item_q['object_names']
        
        # new_programs_decoder = generate_pairs(new_programs)
        # new_programs_hierarchical_decoder = generate_hierarchical_pairs(new_programs)

        program_text_tokenized = None # TODO: compute program_text_tokenized using new_programs_hierarchical_decoder
        sg = None # TODO: compute sg_datum using self.sg_feature_lookup.query_and_translate(queryID, new_execution_buffer)

        # Preprocess question and answer
        question_tokenzied = ScanNetGQA.TEXT.preprocess(question_text)
        answer_label = None # TODO: convert answer to label

        return (question_id, question_tokenzied, sg, program_text_tokenized, answer_label)
        # return (questionID, question_text_tokenized, sg_datum, program_text_tokenized, short_answer_label)
    
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    test_dataset = ScanNetGQA('test')
    print(test_dataset[0])
    