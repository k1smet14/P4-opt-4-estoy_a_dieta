import yaml
from typing import Any, Dict, Union

def read_yaml(cfg: Union[str, Dict[str, Any]]):
    if not isinstance(cfg, dict):
        with open(cfg) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = cfg
    return config

# nas_config = read_yaml('configs/NAS_block/base.yaml')
# print(nas_config.SpaceToChannel)

def get_block_list(path: str):
    nas_yaml = read_yaml(path)
    max_block = nas_yaml['max_blocks']
    
    config = dict()
    block_config = [dict() for _ in range(nas_yaml['max_blocks'])]
    for block in nas_yaml['blocks'].keys():
        start = nas_yaml['blocks'][block]['start']
        for i in range(start,max_block):
            n_dict = dict()
            for k in nas_yaml['blocks'][block].keys():
                if k=='start': continue
                n_dict[k] = nas_yaml['blocks'][block][k][i-start]
                
            block_config[i][block] = n_dict
            
    for i in range(max_block):
        cate = block_config[i].keys()
        block_config[i]['category'] = cate
        block_config[i]['common'] = dict()    
        block_config[i]['common']['activation'] = nas_yaml['common']['activation']
        for k in ['repeat','out_channels','stride']:
            block_config[i]['common'][k] = nas_yaml['common'][k][i]
            
    if 'SpaceToChannel' in nas_yaml:
        config['SpaceToChannel'] = nas_yaml['SpaceToChannel']
        
    config['Classifier'] = nas_yaml['Classifier']
    config['Blocks'] = block_config
    return config
config = get_block_list('configs/NAS_block/Muxconv.yaml')
print(config)