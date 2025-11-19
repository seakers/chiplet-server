import sys
import matplotlib 
matplotlib.use('agg')

import numpy as np
from scipy.optimize import minimize
from collections import Counter
import os

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
import json
from copy import deepcopy

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

        result_file = self.OUTPUT_DIR + "/points.csv"
        entry = f"{total_exe},{total_energy},{numChips[0]},{numChips[1]},{numChips[2]},{numChips[3]}\n"
        # Check if entry already exists
        exists = False
        try:
            with open(result_file, "r") as f:
                for line in f:
                    if line.strip() == entry.strip():
                        exists = True
                        break
        except FileNotFoundError:
            pass  # File does not exist yet

        if not exists:
            with open(result_file, "a") as f:
                f.write(entry)
                print(f"Summary saved to {result_file}")

            context_file = self.OUTPUT_DIR + "/pointContext/" + f"{numChips[0]}gpu{numChips[1]}attn{numChips[2]}sparse{numChips[3]}conv.json"
            os.makedirs(os.path.dirname(context_file), exist_ok=True)
            with open(context_file, "w") as f:
                jsonData = []
                for ind, result in enumerate(agg_kernel_results):
                    resultCopy = deepcopy(result)
                    resultCopy["kernal_number"] = ind
                    jsonData.append(resultCopy)
                json.dump(jsonData, f, indent=4)
            print(f"Results saved to {context_file}")

        else:
            print(f"Entry already exists in {result_file}, not writing duplicate.")

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

def runGACascade(pop_size=10, n_gen=5, trace="", initial_population=None, return_decisions=False, output_dir=None):
    """
    Run the Genetic Algorithm for Cascades.
    If initial_population is provided, use it as the initial population for the GA.
    If return_decisions is True, return both objectives and decision variables.
    output_dir: if provided, use as OUTPUT_DIR for all file writes (e.g., points.csv)
    """
    import numpy as np
    import os
    # --- Always clear points.csv before starting new GA run ---
    result_dir = output_dir if output_dir is not None else OUTPUT_DIR
    points_file = os.path.join(result_dir, "points.csv")
    with open(points_file, 'w') as f:
        pass  # Truncate the file (empty it)
    if isinstance(trace, dict):
        print("Received composite trace:", trace)
        latency = trace.get("latency", 100)
        energy = trace.get("energy", 200)
        points = np.array([
            [latency, energy],
            [latency * 1.1, energy * 0.9],
            [latency * 0.95, energy * 1.05]
        ])
        if return_decisions:
            # Dummy decision variables (zeros)
            decisions = np.zeros((points.shape[0], 12))
            return {"objectives": points, "decisions": decisions}
        return points
    else:
        if trace == "":
            print("No trace provided.")
            print("Selecting a random trace.")
            trace = "gpt-j-65536-weighted"
        WORKSPACE=sys.path[0]+'/api/Evaluator/cascade/chiplet_model'
        TRACE_DIR=WORKSPACE+'/traces'
        CHIPLET_LIBRARY=WORKSPACE+'/dse/chiplet-library'
        EXPERIMENT_DIR=WORKSPACE+'/dse/experiments/'+trace+'.json'
        if output_dir is not None:
            OUTPUT_DIR = output_dir
        else:
            OUTPUT_DIR=WORKSPACE+'/dse/results'
        traces_available = ["gpt-j-65536-weighted", "gpt-j-1024-weighted", "sd-test", "dnn-test", "resnet50-test"]
        print("experiment being performed: ", EXPERIMENT_DIR)
        print("Pop Size: ", pop_size)
        print("Number of Generations: ", n_gen)
        problem = CascadeProblem(TRACE_DIR, CHIPLET_LIBRARY, EXPERIMENT_DIR, OUTPUT_DIR)
        # Use initial_population if provided, else IntegerRandomSampling
        if initial_population is not None:
            print("Using provided initial population for GA.")
            algorithm = NSGA2(pop_size=pop_size,
                              sampling=initial_population,
                              crossover=SBX(eta=15, prob=0.9, repair=RoundingRepair()),
                              mutation=PM(eta=20, repair=RoundingRepair()))
        else:
            algorithm = NSGA2(pop_size=pop_size,
                              sampling=IntegerRandomSampling(),
                              crossover=SBX(eta=15, prob=0.9, repair=RoundingRepair()),
                              mutation=PM(eta=20, repair=RoundingRepair()))
        res = minimize(problem,
                    algorithm,
                    ("n_gen", n_gen),
                    verbose=True)
        print(res.F)
        
        # Save current population for potential custom point integration
        if output_dir is not None:
            import json
            population_file = os.path.join(output_dir, 'current_population.json')
            current_population = res.X.tolist() if hasattr(res, 'X') else []
            with open(population_file, 'w') as f:
                json.dump(current_population, f)
            print(f"Saved current population to {population_file}")
        
        if return_decisions:
            return {"objectives": res.F, "decisions": res.X}
        return res.F

def runSingleCascade(chiplets = {"Attention": 3, "Convolution": 3, "GPU": 3, "Sparse": 3}, trace="", save_to_csv=True):
    """
    Run a single instance of the Cascade model.
    """
    print(f"runSingleCascade called with chiplets: {chiplets}, trace: {trace}, save_to_csv: {save_to_csv}")
    
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
    print(f"Desired chiplets array: {desired_chiplets}")
    print(f"Total chiplets in array: {len(desired_chiplets)}")
    
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

    # Only save to CSV if save_to_csv is True (for genetic algorithm points, not custom designs)
    if save_to_csv:
        result_file = OUTPUT_DIR + "/points.csv"
        entry = f"{total_exe},{total_energy},{chiplets['GPU']},{chiplets['Attention']},{chiplets['Sparse']},{chiplets['Convolution']}\n"
        # Check if entry already exists
        exists = False
        try:
            with open(result_file, "r") as f:
                for line in f:
                    if line.strip() == entry.strip():
                        exists = True
                        break
        except FileNotFoundError:
            pass  # File does not exist yet

        if not exists:
            with open(result_file, "a") as f:
                f.write(entry)
                print(f"Summary saved to {result_file}")

            context_file = OUTPUT_DIR + "/pointContext/" + f"{chiplets['GPU']}gpu{chiplets['Attention']}attn{chiplets['Sparse']}sparse{chiplets['Convolution']}conv.json"
            os.makedirs(os.path.dirname(context_file), exist_ok=True)
            with open(context_file, "w") as f:
                jsonData = []
                for ind, result in enumerate(agg_kernel_results):
                    resultCopy = deepcopy(result)
                    resultCopy["kernal_number"] = ind
                    jsonData.append(resultCopy)
                json.dump(jsonData, f, indent=4)
            print(f"Results saved to {context_file}")

        else:
            print(f"Entry already exists in {result_file}, not writing duplicate.")
    else:
        print("Skipping CSV save for custom design evaluation")

    return float(total_exe), float(total_energy)

# if __name__ == "__main__":
    # runGACascade(pop_size=5, n_gen=5)
    # execTime, energy = runSingleCascade(chiplets={"Attention": 3, "Convolution": 3, "GPU": 3, "Sparse": 3}, trace="gpt-j-65536-weighted")