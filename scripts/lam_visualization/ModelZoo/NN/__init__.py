import os
import torch

MODEL_DIR = 'ModelZoo/models'


NN_LIST = [
    'CARN',
    'IMDN',
    'PCEVA',
]


MODEL_LIST = {
    'CARN': {
        'Base': 'CARN_7400.pth',
    },
    'IMDN': {
        'Base': 'IMDN_x4.pth',
    },
    'PCEVA': {
        'Base': 'PCEVA_L_x4.pth',
    },
}

def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))


def get_model(model_name, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    if model_name.split('-')[0] in NN_LIST:

        if model_name == 'CARN':
            from .CARN.carn import CARNet
            net = CARNet(factor=factor, num_channels=num_channels)
            
        elif model_name == 'IMDN':
            from .NN.imdn import IMDN
            net = IMDN(upscale=factor)
        
        elif model_name == 'PCEVA':
            from .NN.PCEVA import PCEVA
            net = PCEVA(channels=80,sub_channels= [64, 48, 32],  zip_channels=16,  blocks= 4,  scale= 4,  pre_conv= True)
        else:
            raise NotImplementedError()

        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()


def load_model(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    net = get_model(model_name)
    state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')
    state_dict = torch.load(state_dict_path, map_location='cpu')
    new_state_dict = {}
    if model_name in ['IMDN']:
        for k,v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(state_dict)
    return net




