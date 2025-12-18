import os
import argparse

def parse_args(): 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--model-name", type=str, default=None, choices=[None, "llama3-8b", "llama3-70b", "llama3-405b", "llama4-maverick", "llama4-scout"])
    parser.add_argument("--from-pretrained", type=str, default="False")
    parser.add_argument("--weight-dir", type=str, default=None)
    parser.add_argument("--w-dtype", type=float, default=0.5)
    parser.add_argument("--kv-dtype", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--regenerate-output", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--validate-output-ids", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--validate-pistil-intermediates", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--transactional-sim", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--verbose", type=int, default=0)
    
    parser.add_argument("--sim-num-layers", type=int, default=-1)
    parser.add_argument("--sim-batch-size", type=int, default=1) # total batch size for potentially 1 prefill plus remaining decode
    parser.add_argument("--sim-prefill-batch", type=str, default="False", choices=["True", "False"]) # 0 means no prefill, and 1 means there's a prefill 
    parser.add_argument("--sim-prefill-chunk-size", type=int, default=0) # how large of a prefill chunk do you compute?
    parser.add_argument("--sim-prefill-cached", type=int, default=0) # how much of the prefill is already computed and cached? Could just use sim-kv-cache? But this gives more flexability 

    parser.add_argument("--model-cfg", type=str, default=None)
    parser.add_argument("--sys-cfg", type=str, default=None)
    parser.add_argument("--trace-file", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="./trace-results")
    parser.add_argument("--inference", type=str, default="True", choices=["True", "False"])
    
    parser.add_argument("--sim-kv-cache", type=int, default=42) # what is captured in the trace - decode batches - can be treated like average kv_cache across all batches
    parser.add_argument("--kv-cache", type=int, default=1024) # what is run on the system
    parser.add_argument("--sim-vocab-size", type=int, default=128000) # what is run on the system
    parser.add_argument("--sim-lm-head", type=str, default="False", choices=["True", "False"]) # what is run on the system
    
    # system override parameters
    parser.add_argument("--sim-num-chiplets", type=int, default=None)  
    
    # plot parameters
    parser.add_argument("--plot-exe", type=str, default="False") # what is run on the system
    
    args = parser.parse_args()

    args.from_pretrained                    = True if args.from_pretrained == "True" else False
    args.regenerate_output                  = True if args.regenerate_output == "True" else False
    args.validate_output_ids                = True if args.validate_output_ids == "True" else False
    args.validate_pistil_intermediates      = True if args.validate_pistil_intermediates == "True" else False
    args.transactional_sim                  = True if args.transactional_sim == "True" else False
    args.inference                          = True if args.inference == "True" else False
    args.sim_lm_head                        = True if args.sim_lm_head == "True" else False
    args.norm_dtype                         = 2 if args.model_name == "gemma2-2B" else args.w_dtype
    args.plot_exe                           = True if args.plot_exe == "True" else False
    args.sim_prefill_batch                  = True if args.sim_prefill_batch == "True" else False
    args.real_batch_size                    = args.sim_batch_size

    if args.sim_batch_size == 0:
        print("Error: Batch size must be greater than 0")
        exit()

    if args.sim_prefill_batch:
        og_batch_size = args.sim_batch_size
        args.sim_batch_size = args.sim_batch_size - 1 + args.sim_prefill_chunk_size
        print("[arg-parser] Using Prefill Batch!")
        print(f"\tbatch-size {og_batch_size} -> batch+chunked {args.sim_batch_size}")
        print(f"\tPrefill:\t{args.sim_prefill_chunk_size} + KV$ {args.sim_prefill_cached}")
        print(f"\tDecode: \t{og_batch_size-1} + KV$ {args.sim_kv_cache}")

    if args.model_cfg != None:
        if not os.path.exists(args.model_cfg):
            print("Error: Model Config path does not exist: %s" % args.model_cfg)
            exit()

    if args.sys_cfg != None:
        if not os.path.exists(args.sys_cfg):
            print("Error: Sys Config path does not exist: %s" % args.sys_cfg)
            exit()
    
    import json
    with open(args.sys_cfg, 'r') as f:
        sys_cfg = json.load(f)
    
    # override sim-num-chiplets
    if args.sim_num_chiplets == None:
        args.sim_num_chiplets = sys_cfg["NUM_CHIPLETS"]
        

    return args
    
    
