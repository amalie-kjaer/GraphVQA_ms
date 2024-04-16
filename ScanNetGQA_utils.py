from configparser import ConfigParser

def load_config(config_path='/cluster/project/cvg/students/akjaer/GraphVQA/ScanNetGQA_config.yaml'):
    config = ConfigParser()
    config.read(config_path)
    return config