
class HBLC:
    def __init__(self, NAME="HBM3e", BANK_GROUPS=4, CH_PER_LAYER=4, RANKS=4, FRAC_BANK_CAP=1.0, COST_PER_GB_HBM=10):
        # HBM Parameters
        self.NAME               = NAME
        self.BANK_GROUPS        = BANK_GROUPS
        self.CH_PER_LAYER       = CH_PER_LAYER
        self.RANKS              = RANKS
        self.FRAC_BANK_CAP      = FRAC_BANK_CAP

        # Physical Dimensions (mm)
        self.BANK_DEPTH_FULL    = 1.25
        self.BANK_WIDTH         = 0.37
        self.TSV_DEPTH          = 0.34
        self.CH_WIDTH           = 2.5
        self.CAP_PER_BANK       = 24

        # Energy 
        self.PJ_PER_MM          = 0.2
        self.PJ_PER_TSV         = 0.148
        self.PJ_PER_IO          = 0.25
        self.PJ_PER_ACT         = 0.18

        # FRAC_BANK_CAP should influence ACT energy... not sure how yet...
        # Less row energy, fewer control bits, etc.  
        self.COST_PER_GB_HBM    = COST_PER_GB_HBM
        
        if self.BANK_GROUPS < 1 or self.CH_PER_LAYER < 1 or self.RANKS < 1:
            raise ValueError("BANK_GROUPS, CH_PER_LAYER, and RANKS must be at least 1.")
        
    def print_report(self):
        cap         = self.calculate_capacity()
        bw         = self.calculate_bandwidth()
        cap         = self.calculate_capacity()
        pj_per_b    = self.calculate_energy()
        cost        = self.calculate_cost()

        print("########################################")
        print("HBM-Like Mem")
        print("\tName: \t\t%s" % self.NAME)
        print("\tCapacity: \t%0.2f GB" % cap)
        print("\tBandwidth: \t%0.2f GB/s" % bw)
        print("\tBW/Cap: \t%0.2f" % (bw/cap))
        print("\tEnergy: \t%0.2f pJ/b" % (pj_per_b["total"]))
        print("\t\tIO: \t%0.2f pJ/b" % (pj_per_b["io"]))
        print("\t\tTSV: \t%0.2f pJ/b" % (pj_per_b["tsvs"]))
        print("\t\tMOV: \t%0.2f pJ/b" % (pj_per_b["mov-mem"]))
        print("\t\tACT: \t%0.2f pJ/b" % (pj_per_b["act"])) 
        print("\tCost: \t\t$%0.2f" % (cost))
        print("\tCost Per GB: \t$%0.2f" % (cost/cap))
        print("########################################")

    def mem_bytes_per_ns(self):
        return 32

    # enforcing each channel to have 2-pchannels but really that just means 2x channels since they're used fully independent
    def calculate_capacity(self):
        return (self.CAP_PER_BANK * self.FRAC_BANK_CAP * 4) * self.BANK_GROUPS * (self.CH_PER_LAYER * 2) * 4 * self.RANKS / 1024 # GB capacity - min 4 layers to achieve bw
    
    def calculate_capacity_per_ch(self):
        return (self.CAP_PER_BANK * self.FRAC_BANK_CAP * 4) * self.BANK_GROUPS / 1024 # GB capacity - not multiplying by 2 for ch_per_layer bc we want the sudo p-ch capacity

    def calculate_bandwidth(self):
        return (self.CH_PER_LAYER * 2) * 4 * self.mem_bytes_per_ns() # this number needs talign with the pistil-sys-64-chiplets

    def calculate_energy(self):
        self.BG_DEPTH = 2*self.BANK_DEPTH_FULL * self.FRAC_BANK_CAP
        
        if self.BANK_GROUPS == 1:
            AVG_DISTANCE = 2*(self.BANK_WIDTH/2) + ((self.BG_DEPTH/2) + self.TSV_DEPTH)
        elif self.BANK_GROUPS == 2:
            AVG_DISTANCE = 2*(self.BANK_WIDTH/2) + ((self.BG_DEPTH/2) + (self.TSV_DEPTH/2)) + (self.BG_DEPTH + self.TSV_DEPTH)
        elif self.BANK_GROUPS == 3:
            AVG_DISTANCE = 2*(self.BANK_WIDTH/2) + ((2*self.BG_DEPTH/2) + (self.TSV_DEPTH/2))*2/3 + ((self.BG_DEPTH/2) + (self.TSV_DEPTH/2))*1/3 + (self.BG_DEPTH+self.TSV_DEPTH)
        else:
            AVG_DISTANCE = 2*(self.BANK_WIDTH/2) + ((2*self.BG_DEPTH/2) + (self.TSV_DEPTH/2)) + (2*self.BG_DEPTH+self.TSV_DEPTH)
            
        PJ_TSVS     = self.PJ_PER_TSV * ((self.RANKS*4)/2+1)
        PJ_MM       = AVG_DISTANCE * self.PJ_PER_MM
        PJ_PER_BIT  = (PJ_MM + PJ_TSVS) + self.PJ_PER_IO + self.PJ_PER_ACT  # * 0.8 + (self.PJ_PER_ACT * .2 * self.FRAC_BANK_CAP)

        pj_per_b = {}
        pj_per_b["total"]   = PJ_PER_BIT
        pj_per_b["io"]      = self.PJ_PER_IO
        pj_per_b["tsvs"]    = PJ_TSVS
        pj_per_b["mov-mem"] = PJ_MM
        pj_per_b["act"]     = self.PJ_PER_ACT

        return pj_per_b

    def calculate_cost(self):
        # calculate cost using original HBM cost of $12/G * new SI area / total SI area
        AREA_HBM3 = 10.0*10.68*(4*4) + 11*11 
        frac_base_overhead_x = 11/10
        frac_base_overhead_y = 11/10.68

        self.BG_DEPTH = 2*self.BANK_DEPTH_FULL * self.FRAC_BANK_CAP

        if self.BANK_GROUPS == 1:
            DEPTH = self.BG_DEPTH + self.TSV_DEPTH
        elif self.BANK_GROUPS == 2:
            DEPTH = 2*(self.BG_DEPTH + self.TSV_DEPTH)
        elif self.BANK_GROUPS == 3:
            DEPTH = 3*self.BG_DEPTH + 2*self.TSV_DEPTH
        else:
            DEPTH = 4*self.BG_DEPTH + 2*self.TSV_DEPTH

        WIDTH = (self.CH_PER_LAYER * self.CH_WIDTH)
        AREA_NEW = WIDTH * DEPTH * 4 * self.RANKS + WIDTH*frac_base_overhead_x * DEPTH*frac_base_overhead_y
        cost = (self.COST_PER_GB_HBM * 48) * AREA_NEW / AREA_HBM3
        return cost
