from dataclasses import dataclass

@dataclass
class Hparams:
    ### Paths ###
    data_path:str = '/'
    model_path:str = '/statedict'

    ### Model Parameters ###
    conv_layers:int = 5 #currently unused
    kernel_size:int = 5
    batch_size:int = 2000
    num_workers:int = 0 #data is pushed to GPU pre batching, using mutiple workers after the fact causes errors
    learn_rate:float = 1e-3

    ### Training ###
    epochs:int = 3


    #TODO: create methods to save to and load from JSON file, will be used to save hparams with model versions.
