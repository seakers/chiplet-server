import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class HBLC:
    def __init__(self, NAME="HBM3e", BANKS_PER_GROUP=4, CH_PER_LAYER=4, RANKS=4, FRAC_BANK_CAP=1.0, COST_PER_GB_HBM=10):
        # HBM Parameters
        self.NAME               = NAME
        self.BANKS_PER_GROUP    = BANKS_PER_GROUP
        self.CH_PER_LAYER       = CH_PER_LAYER
        self.LAYERS_PER_RANK    = 4
        self.RANKS              = RANKS
        self.FRAC_BANK_CAP      = FRAC_BANK_CAP

        # Physical Dimensions (mm)
        # Heights
        self.BANK_HEIGHT        = 0.5 * self.FRAC_BANK_CAP + .05
        self.Y_CTRL_HEIGHT      = 0.2
        self.POWER_TSV_HEIGHT   = 0.15
        self.IO_TSV_HEIGHT      = 0.35

        # Widths
        self.BANK_WIDTH         = 1.12
        self.X_CTRL_WIDTH       = 0.34

        self.CH_WIDTH           = 2.6
        self.CAP_PER_BANK       = 24        # MB

        # Energy 
        self.PJ_PER_MM          = 0.2
        self.PJ_PER_TSV         = 0.148
        self.PJ_PER_IO          = 0.25
        self.PJ_PER_ACT         = 0.18

        # FRAC_BANK_CAP should influence ACT energy... not sure how yet...
        # Less row energy, fewer control bits, etc.  
        self.COST_PER_GB_HBM    = COST_PER_GB_HBM
        
        if self.BANKS_PER_GROUP < 1 or self.CH_PER_LAYER < 1 or self.RANKS < 1:
            raise ValueError("BANKS_PER_GROUP, CH_PER_LAYER, and RANKS must be at least 1.")
        
    def print_report(self):
        cap         = self.calculate_capacity()
        bw         = self.calculate_bandwidth()
        cap         = self.calculate_capacity()
        pj_per_b    = self.calculate_energy()
        cost        = self.calculate_cost()

        print("####################################")
        print("HBM-Like Mem")
        print("\tName: \t\t%s" % self.NAME)
        print("\tCapacity: \t%0.2f GB" % cap)
        print("\tBandwidth: \t%0.2f GB/s" % bw)
        print("\tBW/Cap: \t%0.2f" % (bw/cap))
        print("\tEnergy: \t%0.2f pJ/b" % (pj_per_b["total"]))
        print("\t\tIO: \t%0.2f pJ/b" % (pj_per_b["io"]))
        print("\t\tTSV: \t%0.2f pJ/b" % (pj_per_b["tsvs"]))
        print("\t\tMOV: \t%0.2f pJ/b" % (pj_per_b["mov"]))
        print("\t\tACT: \t%0.2f pJ/b" % (pj_per_b["act"])) 
        print("\tCost: \t\t$%0.2f" % (cost))
        print("\tCost Per GB: \t$%0.2f" % (cost/cap))
        print("\tCost Per GB/s: \t$%0.2f" % (cost/bw))
        print("####################################")

    def calculate_capacity(self):
        CHANNEL_CAP = (self.CAP_PER_BANK * self.FRAC_BANK_CAP * self.BANKS_PER_GROUP) * 4
        return CHANNEL_CAP * (self.CH_PER_LAYER * 2) * self.LAYERS_PER_RANK * self.RANKS / 1024
    
    def calculate_capacity_per_ch(self):
        return (self.CAP_PER_BANK * self.FRAC_BANK_CAP * self.BANKS_PER_GROUP) * 4 / 1024 # GB capacity - not multiplying by 2 for ch_per_layer bc we want the sudo p-ch capacity

    def calculate_bandwidth(self):
        return (self.CH_PER_LAYER * 2) * 4 * 32

    def mem_bytes_per_ns(self):
        return 32

    def calculate_energy(self):
        # self.BANKS_PER_GROUP, self.RANKS, self.FRAC_BANK_CAP
        BOT_OVER_TOP = self.IO_TSV_HEIGHT + self.BANKS_PER_GROUP*self.BANK_HEIGHT + self.Y_CTRL_HEIGHT + self.POWER_TSV_HEIGHT
        if self.BANKS_PER_GROUP == 1:
            AVG_DISTANCE_TOP_BG = self.IO_TSV_HEIGHT + self.BANK_HEIGHT / 2 + self.Y_CTRL_HEIGHT
            AVG_DISTANCE_BOT_BG = BOT_OVER_TOP + self.BANK_HEIGHT / 2 + self.Y_CTRL_HEIGHT
        if self.BANKS_PER_GROUP == 2:
            AVG_DISTANCE_TOP_BG = self.IO_TSV_HEIGHT + self.BANK_HEIGHT + self.Y_CTRL_HEIGHT/2 + self.Y_CTRL_HEIGHT/2 + self.BANK_HEIGHT / 2
            AVG_DISTANCE_BOT_BG = BOT_OVER_TOP + self.BANK_HEIGHT + self.Y_CTRL_HEIGHT/2 + self.Y_CTRL_HEIGHT/2 + self.BANK_HEIGHT / 2
        if self.BANKS_PER_GROUP == 3:
            AVG_DISTANCE_TOP_BG = self.IO_TSV_HEIGHT + self.BANK_HEIGHT + self.Y_CTRL_HEIGHT/2 + self.Y_CTRL_HEIGHT/2 + (self.BANK_HEIGHT/2 + self.BANK_HEIGHT/2 + 1.5*self.BANK_HEIGHT) / 3
            AVG_DISTANCE_BOT_BG = BOT_OVER_TOP + self.BANK_HEIGHT + self.Y_CTRL_HEIGHT/2 + self.Y_CTRL_HEIGHT/2 + (self.BANK_HEIGHT/2 + self.BANK_HEIGHT/2 + 1.5*self.BANK_HEIGHT) / 3
        if self.BANKS_PER_GROUP == 4:
            AVG_DISTANCE_TOP_BG = self.IO_TSV_HEIGHT + 2*self.BANK_HEIGHT + self.Y_CTRL_HEIGHT/2 + self.Y_CTRL_HEIGHT/2 + self.BANK_HEIGHT
            AVG_DISTANCE_BOT_BG = BOT_OVER_TOP + 2*self.BANK_HEIGHT + self.Y_CTRL_HEIGHT/2 + self.Y_CTRL_HEIGHT/2 + self.BANK_HEIGHT

        DIST_TSV_TO_IO      = self.IO_TSV_HEIGHT + self.POWER_TSV_HEIGHT + self.Y_CTRL_HEIGHT*2 + self.BANK_HEIGHT * self.BANKS_PER_GROUP * 2
        AVG_DISTANCE = (AVG_DISTANCE_TOP_BG + AVG_DISTANCE_BOT_BG) / 2 + DIST_TSV_TO_IO

        # AVG_TSVS    = (sum(np.arange(1, self.RANKS*4+1))/(self.RANKS*4))
        AVG_TSVS    = (self.RANKS*4+1)/2
        PJ_TSVS     = self.PJ_PER_TSV * AVG_TSVS
        PJ_MM       = AVG_DISTANCE * self.PJ_PER_MM
        PJ_PER_BIT  = (PJ_MM + PJ_TSVS) + self.PJ_PER_IO + self.PJ_PER_ACT

        pj_per_b = {}
        pj_per_b["total"]   = PJ_PER_BIT
        pj_per_b["io"]      = self.PJ_PER_IO
        pj_per_b["tsvs"]    = PJ_TSVS
        pj_per_b["mov"]     = PJ_MM
        pj_per_b["act"]     = self.PJ_PER_ACT

        return pj_per_b

    def calculate_cost(self):
        # calculate cost using original HBM cost of $12/G * new SI area / total SI area
        AREA_HBM3 = 10.4*10.6*(4*4) + 11*11 
        frac_base_overhead_x = 11/10
        frac_base_overhead_y = 11/10.68
        
        HEIGHT = 2*(self.IO_TSV_HEIGHT + 2*(self.BANKS_PER_GROUP*self.BANK_HEIGHT + self.Y_CTRL_HEIGHT) + self.POWER_TSV_HEIGHT)
        WIDTH = (self.CH_PER_LAYER * self.CH_WIDTH)

        AREA_NEW = WIDTH * HEIGHT * 4 * self.RANKS   +   WIDTH*frac_base_overhead_x * HEIGHT*frac_base_overhead_y
        cost = (self.COST_PER_GB_HBM * 48) * AREA_NEW / AREA_HBM3
        return cost
