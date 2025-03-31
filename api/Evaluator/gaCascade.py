import sys
import matplotlib 
matplotlib.use('agg')

import numpy as np
from scipy.optimize import minimize
from collections import Counter

from api.Evaluator.cascade.chiplet_model.dse.lib.plot_results import PlotResults
from api.Evaluator.cascade.chiplet_model.dse.lib.chiplet_system import ChipletSystem
from api.Evaluator.cascade.chiplet_model.dse.lib.trace_parser import TraceParser

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

class CascadeProblem(ElementwiseProblem):
    def __init__(self, TRACE_DIR, CHIPLET_LIBRARY, EXPERIMENT_DIR, OUTPUT_DIR):
        super().__init__(n_var=12, 
                         n_obj=2, 
                         n_constr=0, 
                         xl=0, 
                         xu=3,
                         vtype=int)
        self.TRACE_DIR = TRACE_DIR
        self.CHIPLET_LIBRARY = CHIPLET_LIBRARY
        self.EXPERIMENT_DIR = EXPERIMENT_DIR
        self.OUTPUT_DIR = OUTPUT_DIR

        self.tp = TraceParser(self.TRACE_DIR, self.EXPERIMENT_DIR)
    
    def _evaluate(self, x, out, *args, **kwargs):
        numChips = Counter(x)
        desired_chiplets = ["gpu"] * numChips[0] + ["atten"] * numChips[1] + ["sparse"] * numChips[2] + ["conv"] * numChips[3]
        numChannels = 16 # can be changed to be a variable
        agg_kernel_results = []

        for TRACE_ID in range(len(self.tp.all_traces)):
            self.tp.all_traces[TRACE_ID].print_line()
            print("Working on Trace %i" % TRACE_ID)

            # create SoC and add chiplets to SoC
            cs = ChipletSystem(self.CHIPLET_LIBRARY, verbose=0)
            cs.configure_system(desired_chiplets, self.tp.optimization_goal)
            cs.init_system_bandwidth(bw_per_channel=64, num_channels=numChannels)
            # valid = cs.check_valid_system(self.tp.get_trace(TRACE_ID))
            # if not valid == -1:
            kernel_results = cs.characterize_workload(self.tp.get_trace(TRACE_ID), cut_dim="batch" if self.tp.all_traces[TRACE_ID].get_model() == "dnn" else "weights", dtype=2)
            agg_kernel_results += kernel_results * self.tp.all_traces[TRACE_ID].weighted_score # sudo run the workload "weighted_score" times

        total_exe, total_energy = self.getTimeAndEnergy(agg_kernel_results, cs.get_num_chiplets())
        # if total_exe == 0 or total_energy == 0:
        #     out["F"] = [10e6, 10e6]
        # else:

        context_file = self.OUTPUT_DIR + "/pointContext/" + f"{numChips[0]}gpu{numChips[1]}attn{numChips[2]}sparse{numChips[3]}conv.txt"
        with open(context_file, "w") as f:
            for result in agg_kernel_results:
                result_copy = {key: value for key, value in result.items() if key != "chiplets"}
                result_copy["total"] = {key: f"{value:.1e}" for key, value in result_copy["total"].items()}
                f.write(str(result_copy) + "\n")
        print(f"Results saved to {context_file}")

        result_file = self.OUTPUT_DIR + "/points.csv"
        with open(result_file, "a") as f:
            f.write(f"{total_exe},{total_energy},{numChips[0]},{numChips[1]},{numChips[2]},{numChips[3]}\n")
        print(f"Summary saved to {result_file}")

        out["F"] = [total_exe, total_energy]

    def getTimeAndEnergy(self, kernel_results, num_chiplets):
        kernel_names = [] 
        kernel_exe = []
        kernel_energy = []
        kernel_work = {}
        kernel_breakdown = {}
        chiplet_names = []
        for chiplet_id in range(num_chiplets):
            kernel_work[chiplet_id] = []
            chiplet_names.append("")

        for kernel in kernel_results:
            for chiplet_id in range(num_chiplets):
                if chiplet_id in kernel["chiplets"].keys(): # chiplet participated 
                    kernel_work[chiplet_id].append(kernel["chiplets"][chiplet_id]["work"])
                    chiplet_names[chiplet_id] = kernel["chiplets"][chiplet_id]["name"]
                else: # chiplet did not participate
                    kernel_work[chiplet_id].append(0)
            kernel_exe.append(kernel["total"]["exe_time"])
            kernel_energy.append(kernel["total"]["energy"] )

            if kernel["name"] not in kernel_breakdown:
                kernel_breakdown[kernel["name"]] = 0
            kernel_breakdown[kernel["name"]] += kernel["total"]["exe_time"]
            kernel_names.append(kernel["name"])
        
        # normalize kernel_work
        total_exe = sum(kernel_exe)
        total_energy = sum(kernel_energy)

        kernel_work_agg = []
        for chiplet_id in range(num_chiplets):
            kernel_work_agg.append(np.dot(np.array(kernel_work[chiplet_id]), np.array(kernel_exe)/total_exe))
        
        # print(f"{len(kernel_work_agg)} frac work per chiplet: [", end=" ")
        # for frac_work in kernel_work_agg:
        #     print("%0.2f%%" % (frac_work*100), end=" ")
        # print("]")
        print("Total Time: %0.5fms" % (total_exe*1000))
        print("Total Energy: %0.5fmJ" % (total_energy*10**3))

        # # Write outputs to a file
        # outlist = [total_exe*1000, total_energy*10**3]
        # with open(self.OUTPUT_DIR + "/points.txt", "w", newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(outlist)

        return float(total_exe*1000), float(total_energy*10**3)

def runGACascade(pop_size=10, n_gen=5, trace=""):
    """
    Run the Genetic Algorithm for Cascades.
    """
    if trace == "":
        print("No trace provided.")
        print("Selecting a random trace.")
        trace = "gpt-j-65536-weighted"
    # WORKSPACE='chiplet-server/chiplet-model'
    WORKSPACE=sys.path[0]+'/api/Evaluator/cascade/chiplet_model'
    TRACE_DIR=WORKSPACE+'/traces'
    CHIPLET_LIBRARY=WORKSPACE+'/dse/chiplet-library'
    EXPERIMENT_DIR=WORKSPACE+'/dse/experiments/'+trace+'.json' # gpt-j-65536-weighted.json
    OUTPUT_DIR=WORKSPACE+'/dse/results'

    traces_available = ["gpt-j-65536-weighted", "gpt-j-1024-weighted", "sd-test", "dnn-test", "resnet50-test"]

    print("experiment being performed: ", EXPERIMENT_DIR)

    ###################################################
    # hyperparameters for chiplet selection
    GPU     = 3
    ATTEN   = 3
    SPARSE  = 3
    CONV    = 3
    NUM_CHANNELS = 16   # Max number of Memory Channels
    ###################################################

    print("Pop Size: ", pop_size)
    print("Number of Generations: ", n_gen)

    problem = CascadeProblem(TRACE_DIR, CHIPLET_LIBRARY, EXPERIMENT_DIR, OUTPUT_DIR)

    algorithm = NSGA2(pop_size=pop_size,
                      sampling=IntegerRandomSampling(),
                      crossover=SBX(eta=15, prob=0.9, repair=RoundingRepair()),
                      mutation=PM(eta=20, repair=RoundingRepair()))

    res = minimize(problem,
                algorithm,
                ("n_gen", n_gen),
                verbose=True)

    print(res.F)

    return res.F

def runSingleCascade(chiplets = {"Attention": 3, "Convolution": 3, "GPU": 3, "Sparse": 3}, trace=""):
    """
    Run a single instance of the Cascade model.
    """
    if trace == "":
        print("No trace provided.")
        print("Selecting a random trace.")
        trace = "gpt-j-65536-weighted"
    WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
    TRACE_DIR = WORKSPACE + '/traces'
    CHIPLET_LIBRARY = WORKSPACE + '/dse/chiplet-library'
    EXPERIMENT_DIR = WORKSPACE + '/dse/experiments/' + trace + '.json'
    OUTPUT_DIR = WORKSPACE + '/dse/results'

    print("Running single cascade for trace: ", EXPERIMENT_DIR)

    TRACE_DIR = TRACE_DIR
    CHIPLET_LIBRARY = CHIPLET_LIBRARY
    EXPERIMENT_DIR = EXPERIMENT_DIR
    OUTPUT_DIR = OUTPUT_DIR

    tp = TraceParser(TRACE_DIR, EXPERIMENT_DIR)

    # desired_chiplets = ["gpu"] * numChips[0] + ["atten"] * numChips[1] + ["sparse"] * numChips[2] + ["conv"] * numChips[3]
    desired_chiplets = ["gpu"] * chiplets["GPU"] + ["atten"] * chiplets["Attention"] + ["sparse"] * chiplets["Sparse"] + ["conv"] * chiplets["Convolution"]
    numChannels = 16 # can be changed to be a variable
    agg_kernel_results = []

    for TRACE_ID in range(len(tp.all_traces)):
        tp.all_traces[TRACE_ID].print_line()
        print("Working on Trace %i" % TRACE_ID)

        # create SoC and add chiplets to SoC
        cs = ChipletSystem(CHIPLET_LIBRARY, verbose=0)
        cs.configure_system(desired_chiplets, tp.optimization_goal)
        cs.init_system_bandwidth(bw_per_channel=64, num_channels=numChannels)
        # valid = cs.check_valid_system(self.tp.get_trace(TRACE_ID))
        # if not valid == -1:
        kernel_results = cs.characterize_workload(tp.get_trace(TRACE_ID), cut_dim="batch" if tp.all_traces[TRACE_ID].get_model() == "dnn" else "weights", dtype=2)
        agg_kernel_results += kernel_results * tp.all_traces[TRACE_ID].weighted_score # sudo run the workload "weighted_score" times


    # total_exe, total_energy = getTimeAndEnergy(agg_kernel_results, cs.get_num_chiplets())
    num_chiplets = cs.get_num_chiplets()

    kernel_names = [] 
    kernel_exe = []
    kernel_energy = []
    kernel_work = {}
    kernel_breakdown = {}
    chiplet_names = []
    for chiplet_id in range(num_chiplets):
        kernel_work[chiplet_id] = []
        chiplet_names.append("")

    for kernel in agg_kernel_results:
        for chiplet_id in range(num_chiplets):
            if chiplet_id in kernel["chiplets"].keys(): # chiplet participated 
                kernel_work[chiplet_id].append(kernel["chiplets"][chiplet_id]["work"])
                chiplet_names[chiplet_id] = kernel["chiplets"][chiplet_id]["name"]
            else: # chiplet did not participate
                kernel_work[chiplet_id].append(0)
        kernel_exe.append(kernel["total"]["exe_time"])
        kernel_energy.append(kernel["total"]["energy"] )

        if kernel["name"] not in kernel_breakdown:
            kernel_breakdown[kernel["name"]] = 0
        kernel_breakdown[kernel["name"]] += kernel["total"]["exe_time"]
        kernel_names.append(kernel["name"])
    
    # normalize kernel_work
    total_exe = sum(kernel_exe)*1000
    total_energy = sum(kernel_energy)*10**3

    print("Total Time: %0.5fms" % (total_exe))
    print("Total Energy: %0.5fmJ" % (total_energy))

    results_file = OUTPUT_DIR + "/pointContext/" + f"{chiplets['GPU']}gpu{chiplets['Attention']}attn{chiplets['Sparse']}sparse{chiplets['Convolution']}conv.txt"
    with open(results_file, "w") as f:
        for result in agg_kernel_results:
            result_copy = {key: value for key, value in result.items() if key != "chiplets"}
            result_copy["total"] = {key: f"{value:.1e}" for key, value in result_copy["total"].items()}
            f.write(str(result_copy) + "\n")
    print(f"Results saved to {results_file}")

    result_file = OUTPUT_DIR + "/points.csv"
    with open(result_file, "a") as f:
        f.write(f"{total_exe},{total_energy},{chiplets['GPU']},{chiplets['Attention']},{chiplets['Sparse']},{chiplets['Convolution']}\n")
    print(f"Summary saved to {result_file}")

    return float(total_exe), float(total_energy)

# if __name__ == "__main__":
    # runGACascade(pop_size=5, n_gen=5)
    # execTime, energy = runSingleCascade(chiplets={"Attention": 3, "Convolution": 3, "GPU": 3, "Sparse": 3}, trace="gpt-j-65536-weighted")