import copy as cp
from utils.arg_parser import *
from model_lib.model_manager import ModelManager
from pistil_lib.garden import Garden
from pistil_transactor.pistil_system_manager import PistilSystemManager
from pistil_transactor.pistil_transactional_simulator import PistilTransactionalSimulator




if __name__ == "__main__":
    args = parse_args()
    
    sys_manager = PistilSystemManager(args)
    
    llm_manager = ModelManager(args)

    garden = Garden(args=args, sys_manager=sys_manager, llm_manager=llm_manager, ignore_cap=True)
    if llm_manager.model_type == "dense":
        garden.shard_model_dense()
    else:
        garden.shard_model_moe()


    pistil_transactor = PistilTransactionalSimulator(sys_manager, llm_manager, garden)
    pistil_transactor.load_trace(args.trace_file)

    all_events = pistil_transactor.run()


    print("Saving Results")
    from visualize.save_results import *
    sys_cfg = sys_manager.sys_cfg
    occ = "%0.2f" % (sys_cfg["CORE"]["MEMORY_BUFFER_B"] / (1024*1024))
    save_results(cp.deepcopy(all_events), sys_manager, llm_manager, args, f"{args.results_dir}/{llm_manager.model_name}-chiplets-{args.sim_num_chiplets}-r-{sys_cfg["MEMORY_CHIPLET"]["RANKS"]}-bg-{sys_cfg["MEMORY_CHIPLET"]["BANK_GROUPS"]}-f-{sys_cfg["MEMORY_CHIPLET"]["FRAC_BANK_CAP"]}-bs-{args.sim_batch_size}-kv-{args.sim_kv_cache}-occ-{occ}.csv")

    from visualize.visualize_transactions import *
    if args.plot_exe:
        print("Plotting Exe Time")
        plot_execution_timeline(cp.deepcopy(all_events), sys_manager, llm_manager, "./results/exe-timeline.png")
    
    
    print("Simulation Finished")


