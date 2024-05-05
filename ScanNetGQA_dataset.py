import numpy as np
import os
import json
import torchtext
import torch_geometric
import torch

from ScanNetGQA_utils import load_config

config = load_config()
dataset_path = config['path']['dataset_path']
datasets = {
    'train': str(),
    'val': str(),
    'test': ['/cluster/project/cvg/students/akjaer/GraphVQA/ScanQA_v1.0_train_scene0000_00.json', '/cluster/project/cvg/students/akjaer/GraphVQA/scene_graph.json']
}

class ScanNetGQA():
    TEXT = torchtext.data.Field(
        sequential=True,
        tokenize="spacy",
        init_token="<start>",
        eos_token="<end>",
        tokenizer_language='en_core_web_sm',
        include_lengths=False,
        batch_first=False
    )
    SG_ENCODING_TEXT = torchtext.data.Field(
        sequential=True, tokenize="spacy",
        init_token="<start>", eos_token="<end>",
        include_lengths=False,
        tokenizer_language='en_core_web_sm',
        batch_first=False
    )
    MAX_EXECUTION_STEP = 5

    def __init__(self, split):
        self.split = split
        # assert split in ...

        # Load question-answer data
        dataset_path = datasets[split][0]
        with open(dataset_path) as infile:
            self.questions_data = json.load(infile)
        dataset_path = datasets[split][1]
        with open(dataset_path) as infile:
            self.sg_data = json.load(infile)

        # Build look-up vocabulary table for scene graph data
        self.build_scene_vocab_lookup()
        self.build_qa_vocab()
        print(f'Finished loading the data, totally {len(self.questions_data)} instances')

    def __getitem__(self, index):
        '''
        QUESTION-ANSWER DATA
        '''
        # Load question data
        q = self.questions_data[index]
        scene_id = q['scene_id']
        question_text = q['question']
        question_id = q['question_id']
        answer_text = q['answers']
        answer_instances = q['object_ids']
        answer_object_names = q['object_names']
        
        # Preprocess question and answer
        question_tokenzied = ScanNetGQA.TEXT.preprocess(question_text)
        answer_tokenized = ScanNetGQA.TEXT.preprocess(answer_text)
        answer_label = answer_text # TODO: convert answer to label

        '''
        SCENE-GRAPH DATA
        '''
        sg = self.sg_data # TODO: Generalize to more scenes (this is just for scene0000_00)
        sg_torchgeo = self.to_torch_geometric_data(sg)

        return (question_id, question_text, question_tokenzied, sg_torchgeo, answer_tokenized, answer_label)
    
    def __len__(self):
        return len(self.questions_data)
    
    def build_scene_vocab_lookup(self):
        def load_str_list(fname):
            with open(fname) as f:
                lines = f.read().splitlines()
            return lines
        
        txt_list = []
        txt_list += load_str_list('ScanRefer_filtered_attributes.txt')
        txt_list += load_str_list('ScanRefer_filtered_objects.txt')
        txt_list += load_str_list('ScanNet_edges.txt')
        txt_list.append("<self>") # add special token for self-connection
        ScanNetGQA.SG_ENCODING_TEXT.build_vocab(txt_list, vectors="glove.6B.300d")

        return ScanNetGQA.SG_ENCODING_TEXT

    def build_qa_vocab(self):
        all_qa_vocab = []
        for q in self.questions_data:
            question_text = q['question']
            answer_text = q['answers']
            question_tokenzied = ScanNetGQA.TEXT.preprocess(question_text)
            answer_tokenized = ScanNetGQA.TEXT.preprocess(answer_text)
            all_qa_vocab.append(question_tokenzied)
            all_qa_vocab.append(answer_tokenized)
        ScanNetGQA.TEXT.build_vocab(all_qa_vocab, vectors="glove.6B.300d")

    def to_torch_geometric_data(self, sg):
        SG_ENCODING_TEXT = ScanNetGQA.SG_ENCODING_TEXT
        MAX_OBJ_TOKEN_LEN = 13 # TODO: check that this makes sense for our case?
        
        # Initialize node features, edge topology (adjacency) and edge feature list for graph representation
        node_feature_list = []
        edge_feature_list = [] # [[from, to], ...]
        edge_topology_list = []
        added_sym_edge_list = []

        # Looping through all nodes/objects in the scene graph
        for obj_idx, obj in sg['scene0000_00'].items(): # TODO: generalize to more scenes
            obj_idx = int(obj_idx)
            object_token_arr = np.ones(MAX_OBJ_TOKEN_LEN, dtype=np.int_) * SG_ENCODING_TEXT.vocab.stoi[SG_ENCODING_TEXT.pad_token]
            object_token_arr[0] = SG_ENCODING_TEXT.vocab.stoi[obj['label']] # Comment this out to see the importance of the label
            for attr_idx, attr in enumerate(set(obj['attributes'])): # Comment this out to see the importance of attributes
                object_token_arr[attr_idx + 1] = SG_ENCODING_TEXT.vocab.stoi[attr]
            node_feature_list.append(object_token_arr)

            # Add a self-looping edge
            edge_topology_list.append([obj_idx, obj_idx]) # [from self, to self]
            edge_token_arr = np.array([SG_ENCODING_TEXT.vocab.stoi['<self>']], dtype=np.int_)
            edge_feature_list.append(edge_token_arr)

            # Looping through all relations of the current node
            for rel in obj['relations']:
                edge_topology_list.append([obj_idx, rel['object']])
                edge_token_arr = np.array([SG_ENCODING_TEXT.vocab.stoi[rel["edge_label"]]], dtype=np.int_)
                edge_feature_list.append(edge_token_arr)

        node_feature_list_arr = np.stack(node_feature_list, axis=0)
        x = torch.from_numpy(node_feature_list_arr).long()

        edge_feature_list_arr = np.stack(edge_feature_list, axis=0)
        edge_attr = torch.from_numpy(edge_feature_list_arr).long()

        edge_index = torch.tensor(edge_topology_list, dtype=torch.long)

        datum = torch_geometric.data.Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
        # datum.added_sym_edge = added_sym_edge
        
        return datum
    
    @classmethod
    def indices_to_string(cls, indices, words=False):
        """Convert word indices (torch.Tensor) to sentence (string).
        Args:
            indices: torch.tensor or numpy.array of shape (T) or (T, 1)
            words: boolean, wheter return list of words
        Returns:
            sentence: string type of converted sentence
            words: (optional) list[string] type of words list
        """
        sentence = list()
        for idx in indices:
            word = ScanNetGQA.TEXT.vocab.itos[idx.item()]

            if word in ["<pad>", "<start>"]:
                continue
            if word in ["<end>"]:
                break

            # no needs of space between the special symbols
            if len(sentence) and word in ["'", ".", "?", "!", ","]:
                sentence[-1] += word
            else:
                sentence.append(word)

        if words:
            return " ".join(sentence), sentence
        return " ".join(sentence)

def ScanNetGQA_collate_fn(data):
    """
    The collate_fn function working with pytorch data loader.

    Since we have both torchtext data and pytorch geometric data, and both of them
    require non-defualt data batching behaviour, we therefore create a custom collate fn funciton.

    input: data: is a list of tuples with (question_src, sg_datum, program_trg, ...)
            which is determined by the get item method of the dataset

    output: all fileds in batched format
    """
    question_id, question_text, question_tokenzied, sg_torchgeo, answer_tokenized, answer_label = zip(*data)
    question_text_processed = ScanNetGQA.TEXT.process(question_tokenzied)
    sg_processed = torch_geometric.data.Batch.from_data_list(sg_torchgeo)
    answer_processed = ScanNetGQA.TEXT.process(answer_tokenized)


    return (question_id, question_text, question_text_processed, sg_processed, answer_processed, answer_label)


if __name__ == '__main__':
    test_dataset = ScanNetGQA('test')
    print(test_dataset[0])
    