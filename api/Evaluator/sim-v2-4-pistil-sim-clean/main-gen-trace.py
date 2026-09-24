import os

from utils.arg_parser import *
from model_lib.model_manager import ModelManager
from pistil_lib.garden import Garden
from pistil_transactor.pistil_system_manager import PistilSystemManager
from pistil_transactor.pistil_transactional_simulator import PistilTransactionalSimulator


if __name__ == "__main__":
    args = parse_args()
    if "llama4" in args.model_name:
        args.sim_num_layers = 2 # need 2 layers to capture accurate behavior of llama4 model
    else:
        args.sim_num_layers = 1 # need 1 layer to capture accurate behavior of llama3 model

    #if os.path.exists(args.trace_file + ".pkl"):
    #    print("Trace File Already Exists: ", args.trace_file)
    #    exit()

    sys_manager = PistilSystemManager(args)

    llm_manager = ModelManager(args)

    if args.inference:
        llm_manager.my_model_forward() # generate forward KV$ data

    garden = Garden(args=args, sys_manager=sys_manager, llm_manager=llm_manager, ignore_cap=True)
    pistil_transactor = PistilTransactionalSimulator(sys_manager, llm_manager, garden)
    garden.chiplet_garden[0].pistil_transactor = pistil_transactor # attach the transactional simulator to one of the pistil flowers
    garden.pistil_transactor = pistil_transactor # attach the transactional simulator to the garden as well 
    
    if llm_manager.model_type == "dense":
        garden.shard_model_dense()
        garden.load_model_shards_dense()
    elif llm_manager.model_type == "moe":
        garden.shard_model_moe()
        garden.load_model_shards_moe()    

    garden.sim_model()

    pistil_transactor.save_trace(args.trace_file)



