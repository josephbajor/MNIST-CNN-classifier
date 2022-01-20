from dataclasses import dataclass

@dataclass
class Hparams:

    ### Model Parameters ###
    conv_layers:int = 5 #currently unused
    kernel_size:int = 5
    batch_size:int = 20
    num_workers:int = 0 #data is pushed to GPU pre batching, using mutiple workers after the fact causes errors
    learn_rate:float = 1e-3

    ### Training ###
    epochs:int = 5