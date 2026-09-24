import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Tuple, List


class RPU:
    def __init__(self, 
        # core
        core_width,
        core_height,
        buffer_width,
        mem_buffer_height,
        compute_bus_height,
        tmacs_per_core,
        tmac_compute_height,
        tmac_buffer_height,
        tmac_hp_ops_height,
        net_buffer_height,
        net_bus_height,
        ctrl_height,
        
        # compute chiplet
        num_cores_width,
        num_cores_height,
        ucie_buffer,
        cu_width,
        cu_height,
        mem_shoreline,
        mem_gap,
        
        # hbm-co 
        bank_height,
        bank_width,
        banks_per_group,
        y_ctrl_height,
        tsv_height,
        hbm_co_height,
        hbm_co_width,
    ):
        # core
        self.core_width = core_width
        self.core_height = core_height
        self.buffer_width = buffer_width
        self.mem_buffer_height = mem_buffer_height
        self.compute_bus_height = compute_bus_height
        self.tmacs_per_core = tmacs_per_core
        self.tmac_compute_height = tmac_compute_height
        self.tmac_buffer_height = tmac_buffer_height
        self.tmac_hp_ops_height = tmac_hp_ops_height
        self.net_buffer_height = net_buffer_height
        self.net_bus_height = net_bus_height
        self.ctrl_height = ctrl_height
        
        # compute chiplet
        self.num_cores_width = num_cores_width
        self.num_cores_height = num_cores_height
        self.ucie_buffer = ucie_buffer
        self.cu_width = cu_width
        self.cu_height = cu_height
        self.mem_shoreline = mem_shoreline
        self.mem_gap = mem_gap

        # hbm-co
        self.bank_height = bank_height
        self.bank_width = bank_width
        self.banks_per_group = banks_per_group
        self.y_ctrl_height = y_ctrl_height
        self.tsv_height = tsv_height
        self.hbm_co_height = hbm_co_height 
        self.hbm_co_width = hbm_co_width 
    
    def add_box(self, ax, xy: Tuple[float, float], wh: Tuple[float, float], label: str = "",
                edgecolor: str = "black", facecolor: str = "white", alpha: float = 1.0,
                fontsize: int = 9, rotation: float = 0, ha: str = "center", va: str = "center", textcolor: str = "black"):

        rect = Rectangle(xy, wh[0], wh[1], fill=True, edgecolor=edgecolor, facecolor=facecolor, linewidth=1.0, alpha=alpha)
        ax.add_patch(rect)
        cx = xy[0] + wh[0] / 2
        cy = xy[1] + wh[1] / 2
        ax.text(cx, cy, label, fontsize=fontsize, rotation=rotation, ha=ha, va=va, color=textcolor)
        return rect
            
    def plot_core(self, ax, x_origin, y_origin, orientation=1, text=True):
        # Colors
        BUFFER_COLOR = '#BDD4E5'
        ARBITER_COLOR = '#A2BBB8'
        HP_OPS_COLOR = '#CC8FF2'
        ACT_BUF_COLOR = '#F7E6A6'
        TMAC_COLOR = '#DFC0F3'
        COMPUTE_BUS_COLOR = '#512866'
        STREAM_DECODE_COLOR = '#AFABAB'
        MEM_BUS_COLOR = '#595959'
        NET_BUS_COLOR = '#334133'
        
        OUTLINE_COLOR = '#262626'
        TEXT_COLOR = "white"
        TMAC_LINE_WIDTH = .3
        VEC_HEIGHT_SCALAR = 1/3.5

        FONTSIZE = 9

        # core outline
        y_offset = 0
        self.add_box(ax, [x_origin, y_origin], [self.core_width, orientation*self.core_height], label="" if text else "", facecolor="white", edgecolor=OUTLINE_COLOR)
        
        # ctrl
        self.add_box(ax, [x_origin, y_origin], [self.buffer_width, orientation*self.ctrl_height], label="CTRL" if text else "", facecolor=ARBITER_COLOR, edgecolor=OUTLINE_COLOR)
        self.add_box(ax, [x_origin + self.core_width - self.buffer_width, y_origin], [self.buffer_width, orientation*self.ctrl_height], label="I$" if text else "", facecolor=BUFFER_COLOR, edgecolor=OUTLINE_COLOR)
        self.add_box(ax, [x_origin+self.buffer_width, y_origin], [self.core_width-2*self.buffer_width, orientation*self.ctrl_height], label="BUS" if text else "", facecolor=MEM_BUS_COLOR, edgecolor=OUTLINE_COLOR, textcolor=TEXT_COLOR, rotation=-90)
        y_offset += orientation*self.ctrl_height
        
        # network bus
        self.add_box(ax, [x_origin, y_origin+y_offset], [self.core_width, orientation*self.net_bus_height], label="NETWORK BUS" if text else "", facecolor=NET_BUS_COLOR, edgecolor=OUTLINE_COLOR, textcolor=TEXT_COLOR)
        y_offset += orientation*self.net_bus_height
        
        # network buffer
        self.add_box(ax, [x_origin, y_origin+y_offset], [self.buffer_width, orientation*self.net_buffer_height], label="NET /\nGLOBAL\nBUFFER" if text else "", facecolor=BUFFER_COLOR, edgecolor=OUTLINE_COLOR)
        self.add_box(ax, [x_origin + self.core_width - self.buffer_width, y_origin+y_offset], [self.buffer_width, orientation*self.net_buffer_height], label="NET /\nGLOBAL\nBUFFER" if text else "", facecolor=BUFFER_COLOR, edgecolor=OUTLINE_COLOR)

        # internal network bus
        stacked_tmacs = int((self.tmacs_per_core-2)/2)
        tmac_height = (self.tmac_compute_height + self.tmac_buffer_height + self.tmac_hp_ops_height)
        nb_to_cb_height = tmac_height * stacked_tmacs + self.net_buffer_height
        self.add_box(ax, [x_origin+self.buffer_width, y_origin+y_offset], [self.core_width-2*self.buffer_width, orientation*nb_to_cb_height], label="NET / GLOBAL BUS" if text else "", facecolor=MEM_BUS_COLOR, edgecolor=OUTLINE_COLOR, textcolor=TEXT_COLOR, rotation=-90)
        y_offset += orientation*self.net_buffer_height

        # add tmacs
        for tmac in range(stacked_tmacs):
            self.add_box(ax, [x_origin, y_origin+y_offset], [self.buffer_width, orientation*self.tmac_hp_ops_height], label="HP OPS" if text else "", facecolor=HP_OPS_COLOR, edgecolor=OUTLINE_COLOR)
            self.add_box(ax, [x_origin + self.core_width - self.buffer_width, y_origin+y_offset], [self.buffer_width, orientation*self.tmac_hp_ops_height], label="HP OPS" if text else "", facecolor=HP_OPS_COLOR, edgecolor=OUTLINE_COLOR)
            y_offset += orientation*self.tmac_hp_ops_height
            self.add_box(ax, [x_origin, y_origin+y_offset], [self.buffer_width, orientation*self.tmac_buffer_height], label="ACT/ACC BUF" if text else "", facecolor=ACT_BUF_COLOR, edgecolor=OUTLINE_COLOR)
            self.add_box(ax, [x_origin + self.core_width - self.buffer_width, y_origin+y_offset], [self.buffer_width, orientation*self.tmac_buffer_height], label="ACT/ACC BUF" if text else "", facecolor=ACT_BUF_COLOR, edgecolor=OUTLINE_COLOR)
            y_offset += orientation*self.tmac_buffer_height
            self.add_box(ax, [x_origin, y_origin+y_offset], [self.buffer_width, orientation*self.tmac_compute_height], label="TMAC" if text else "", facecolor=TMAC_COLOR, edgecolor=OUTLINE_COLOR)
            self.add_box(ax, [x_origin + self.core_width - self.buffer_width, y_origin+y_offset], [self.buffer_width, orientation*self.tmac_compute_height], label="TMAC" if text else "", facecolor=TMAC_COLOR, edgecolor=OUTLINE_COLOR)
            for i in range(7):
                # bottom lines
                ax.plot([x_origin + (i+1)*self.buffer_width/8, x_origin + (i+1)*self.buffer_width/8], [y_origin+y_offset, y_origin+y_offset+orientation*(self.tmac_compute_height*VEC_HEIGHT_SCALAR)], "--", linewidth=TMAC_LINE_WIDTH, color="black")
                ax.plot([x_origin + self.core_width - self.buffer_width + (i+1)*self.buffer_width/8, x_origin + self.core_width - self.buffer_width + (i+1)*self.buffer_width/8], [y_origin+y_offset, y_origin+y_offset+orientation*self.tmac_compute_height*VEC_HEIGHT_SCALAR], "--", linewidth=TMAC_LINE_WIDTH, color="black")
                
                # # top lines
                ax.plot([x_origin + (i+1)*self.buffer_width/8, x_origin + (i+1)*self.buffer_width/8], [y_origin+y_offset+orientation*self.tmac_compute_height, y_origin+y_offset+orientation*self.tmac_compute_height-orientation*self.tmac_compute_height*VEC_HEIGHT_SCALAR], "--", linewidth=TMAC_LINE_WIDTH, color="black")
                ax.plot([x_origin + self.core_width - self.buffer_width + (i+1)*self.buffer_width/8, x_origin + self.core_width - self.buffer_width + (i+1)*self.buffer_width/8], [y_origin+y_offset+orientation*self.tmac_compute_height, y_origin+y_offset+orientation*self.tmac_compute_height-orientation*self.tmac_compute_height*VEC_HEIGHT_SCALAR], "--", linewidth=TMAC_LINE_WIDTH, color="black")
            y_offset += orientation*self.tmac_compute_height 

        # compute bus
        self.add_box(ax, [x_origin, y_origin+y_offset], [self.core_width, orientation*self.compute_bus_height], label="COMPUTE BUS" if text else "", facecolor=COMPUTE_BUS_COLOR, edgecolor=OUTLINE_COLOR)
        y_offset += orientation*self.compute_bus_height 

        # stream dq
        self.add_box(ax, [x_origin+self.buffer_width, y_origin+y_offset], [self.core_width-2*self.buffer_width, orientation*tmac_height], label="STREAM DQ" if text else "", facecolor=STREAM_DECODE_COLOR, edgecolor=OUTLINE_COLOR, rotation=-90)

        # top tmacs (need at least 2)
        self.add_box(ax, [x_origin, y_origin+y_offset], [self.buffer_width, orientation*self.tmac_compute_height], label="TMAC" if text else "", facecolor=TMAC_COLOR, edgecolor=OUTLINE_COLOR)
        self.add_box(ax, [x_origin + self.core_width - self.buffer_width, y_origin+y_offset], [self.buffer_width, orientation*self.tmac_compute_height], label="TMAC" if text else "", facecolor=TMAC_COLOR, edgecolor=OUTLINE_COLOR)
        for i in range(7):
            # bottom lines
            ax.plot([x_origin + (i+1)*self.buffer_width/8, x_origin + (i+1)*self.buffer_width/8], [y_origin+y_offset, y_origin+y_offset+orientation*self.tmac_compute_height*VEC_HEIGHT_SCALAR], "--", linewidth=TMAC_LINE_WIDTH, color="black")
            ax.plot([x_origin + self.core_width - self.buffer_width + (i+1)*self.buffer_width/8, x_origin + self.core_width - self.buffer_width + (i+1)*self.buffer_width/8], [y_origin+y_offset, y_origin+y_offset+orientation*self.tmac_compute_height*VEC_HEIGHT_SCALAR], "--", linewidth=TMAC_LINE_WIDTH, color="black")
            
            # top lines
            ax.plot([x_origin + (i+1)*self.buffer_width/8, x_origin + (i+1)*self.buffer_width/8], [y_origin+y_offset+orientation*self.tmac_compute_height, y_origin+y_offset+orientation*self.tmac_compute_height-orientation*self.tmac_compute_height*VEC_HEIGHT_SCALAR], "--", linewidth=TMAC_LINE_WIDTH, color="black")
            ax.plot([x_origin + self.core_width - self.buffer_width + (i+1)*self.buffer_width/8, x_origin + self.core_width - self.buffer_width + (i+1)*self.buffer_width/8], [y_origin+y_offset+orientation*self.tmac_compute_height, y_origin+y_offset+orientation*self.tmac_compute_height-orientation*self.tmac_compute_height*VEC_HEIGHT_SCALAR], "--", linewidth=TMAC_LINE_WIDTH, color="black")
        y_offset += orientation*self.tmac_compute_height 
        self.add_box(ax, [x_origin, y_origin+y_offset], [self.buffer_width, orientation*self.tmac_buffer_height], label="ACT/ACC BUF" if text else "", facecolor=ACT_BUF_COLOR, edgecolor=OUTLINE_COLOR)
        self.add_box(ax, [x_origin + self.core_width - self.buffer_width, y_origin+y_offset], [self.buffer_width, orientation*self.tmac_buffer_height], label="ACT/ACC BUF" if text else "", facecolor=ACT_BUF_COLOR, edgecolor=OUTLINE_COLOR)
        y_offset += orientation*self.tmac_buffer_height
        self.add_box(ax, [x_origin, y_origin+y_offset], [self.buffer_width, orientation*self.tmac_hp_ops_height], label="HP OPS" if text else "", facecolor=HP_OPS_COLOR, edgecolor=OUTLINE_COLOR)
        self.add_box(ax, [x_origin + self.core_width - self.buffer_width, y_origin+y_offset], [self.buffer_width, orientation*self.tmac_hp_ops_height], label="HP OPS" if text else "", facecolor=HP_OPS_COLOR, edgecolor=OUTLINE_COLOR)
        y_offset += orientation*self.tmac_hp_ops_height


        # memory buffer
        self.add_box(ax, [x_origin, y_origin+y_offset], [self.buffer_width, orientation*self.mem_buffer_height], label="MEM\nBUFFER" if text else "", facecolor=BUFFER_COLOR, edgecolor=OUTLINE_COLOR)
        self.add_box(ax, [x_origin + self.core_width - self.buffer_width, y_origin+y_offset], [self.buffer_width, orientation*self.mem_buffer_height], label="MEM\nBUFFER" if text else "", facecolor=BUFFER_COLOR, edgecolor=OUTLINE_COLOR)
        self.add_box(ax, [x_origin+self.buffer_width, y_origin+y_offset], [self.core_width-2*self.buffer_width, orientation*self.mem_buffer_height], label="MEMORY BUS" if text else "", facecolor=MEM_BUS_COLOR, edgecolor=OUTLINE_COLOR, textcolor=TEXT_COLOR, rotation=-90)
        
    def plot_compute_chiplet(self, ax, x_origin, y_origin):
        CU_BACKGROUND = '#D3E3ED'
        ARBITER_COLOR = '#A2BBB8'
        MEM_SHORELINE_COLOR = '#F3D97A'
        OUTLINE_COLOR = '#262626'
        
        self.add_box(ax, [x_origin, y_origin-self.core_height-self.mem_gap], [self.cu_width, self.cu_height], facecolor=CU_BACKGROUND, edgecolor=OUTLINE_COLOR)

        for i in range(self.num_cores_width):
            x_val = x_origin + i*self.core_width + self.ucie_buffer
            self.plot_core(ax, x_val, y_origin, orientation=1, text=False)
            self.plot_core(ax, x_val, y_origin, orientation=-1, text=False) 

        # ucie shoreline
        UCIE_BUFFER_WIDTH = 2/3
        UCIE_HEIGHT = self.core_height / self.num_cores_width
        for i in range(self.num_cores_width):
            # left side
            self.add_box(ax, [x_origin, y_origin+i*UCIE_HEIGHT], [self.ucie_buffer*UCIE_BUFFER_WIDTH, UCIE_HEIGHT], facecolor=ARBITER_COLOR, edgecolor=OUTLINE_COLOR)
            self.add_box(ax, [x_origin, y_origin-i*UCIE_HEIGHT], [self.ucie_buffer*UCIE_BUFFER_WIDTH, -1*UCIE_HEIGHT], facecolor=ARBITER_COLOR, edgecolor=OUTLINE_COLOR)
            
            # right side
            self.add_box(ax, [x_origin+self.core_width * self.num_cores_width + self.ucie_buffer + self.ucie_buffer*(1-UCIE_BUFFER_WIDTH), y_origin+i*UCIE_HEIGHT], [self.ucie_buffer*UCIE_BUFFER_WIDTH, UCIE_HEIGHT], facecolor=ARBITER_COLOR, edgecolor=OUTLINE_COLOR)
            self.add_box(ax, [x_origin+self.core_width * self.num_cores_width + self.ucie_buffer + self.ucie_buffer*(1-UCIE_BUFFER_WIDTH), y_origin-i*UCIE_HEIGHT], [self.ucie_buffer*UCIE_BUFFER_WIDTH, -1*UCIE_HEIGHT], facecolor=ARBITER_COLOR, edgecolor=OUTLINE_COLOR)
        
        MEM_BUFFER_HEIGHT = 2/3
        MEM_WIDTH = self.mem_shoreline / self.num_cores_width
        # mem shoreline
        for i in range(int(self.num_cores_width/2)):
            # top side 
            self.add_box(ax, [x_origin+self.cu_width/2 + i*MEM_WIDTH, y_origin+self.core_height+self.mem_gap], [MEM_WIDTH, -1*self.mem_gap*MEM_BUFFER_HEIGHT], facecolor=MEM_SHORELINE_COLOR, edgecolor=OUTLINE_COLOR)
            self.add_box(ax, [x_origin+self.cu_width/2 - i*MEM_WIDTH, y_origin+self.core_height+self.mem_gap], [-1*MEM_WIDTH, -1*self.mem_gap*MEM_BUFFER_HEIGHT], facecolor=MEM_SHORELINE_COLOR, edgecolor=OUTLINE_COLOR)
            
            # bottom side 
            self.add_box(ax, [x_origin+self.cu_width/2 + i*MEM_WIDTH, y_origin-self.core_height-self.mem_gap], [MEM_WIDTH, self.mem_gap*MEM_BUFFER_HEIGHT], facecolor=MEM_SHORELINE_COLOR, edgecolor=OUTLINE_COLOR)
            self.add_box(ax, [x_origin+self.cu_width/2 - i*MEM_WIDTH, y_origin-self.core_height-self.mem_gap], [-1*MEM_WIDTH, self.mem_gap*MEM_BUFFER_HEIGHT], facecolor=MEM_SHORELINE_COLOR, edgecolor=OUTLINE_COLOR)
    
    def plot_hbm_co(self, ax, x_origin, y_origin, text=True):
        MEMORY_COLOR = '#F3B16F'
        MEMORY_BG_COLOR = '#FBE5CF' # background color
        TSV_COLOR = '#AFABAB'
        TSV_ARRAY_COLOR = '#767171'
        OUTLINE_COLOR = '#262626'

        BUFFER_VIS = 0.02

        self.add_box(ax, [x_origin, y_origin], [self.hbm_co_width, self.hbm_co_height], facecolor=MEMORY_BG_COLOR, edgecolor=OUTLINE_COLOR)


        # plot each bank group
        bank_group_height = self.banks_per_group * self.bank_height + self.y_ctrl_height
        offset = bank_group_height*2+self.tsv_height
        # left and right side of memory
        for lr in range(2):
            # top and bototm 2 bank groups
            for i in range(2):
                if self.banks_per_group == 1:
                    label_ab = f"BANK GROUP {'A' if i == 0 else 'B'}" if text else ""
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_ab, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+self.bank_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.y_ctrl_height-2*BUFFER_VIS], label="Y-CTRL" if text else "", facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                
                if self.banks_per_group == 2:
                    label_ab = f"BANK GROUP {'A' if i == 0 else 'B'}" if text else ""
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_ab, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+self.bank_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.y_ctrl_height-2*BUFFER_VIS], label="Y-CTRL" if text else "", facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+self.bank_height+self.y_ctrl_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_ab, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                
                if self.banks_per_group == 3:
                    label_ab = f"BANK GROUP {'A' if i == 0 else 'B'}" if text else ""
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_ab, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+self.bank_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_ab, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+self.bank_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.y_ctrl_height-2*BUFFER_VIS], label="Y-CTRL" if text else "", facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+self.bank_height*2+self.y_ctrl_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_ab, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                
                if self.banks_per_group == 4:
                    label_ab = f"BANK GROUP {'A' if i == 0 else 'B'}" if text else ""
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_ab, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+self.bank_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_ab, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+self.bank_height*2+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.y_ctrl_height-2*BUFFER_VIS], label="Y-CTRL", facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+self.bank_height*2+self.y_ctrl_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_ab, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+i*bank_group_height+self.bank_height*3+self.y_ctrl_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_ab, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                
                
                # top 2 bank groups
                if self.banks_per_group == 1:
                    label_cd = f"BANK GROUP {'C' if i == 0 else 'D'}" if text else ""
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.y_ctrl_height-2*BUFFER_VIS], label="Y-CTRL" if text else "", facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+self.y_ctrl_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_cd, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                
                if self.banks_per_group == 2:
                    label_cd = f"BANK GROUP {'C' if i == 0 else 'D'}" if text else ""
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_cd, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+self.bank_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.y_ctrl_height-2*BUFFER_VIS], label="Y-CTRL" if text else "", facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+self.bank_height+self.y_ctrl_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_cd, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                
                if self.banks_per_group == 3:
                    label_cd = f"BANK GROUP {'C' if i == 0 else 'D'}" if text else ""
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_cd, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+self.bank_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.y_ctrl_height-2*BUFFER_VIS], label="Y-CTRL" if text else "", facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+self.bank_height+self.y_ctrl_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_cd, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+self.bank_height*2+self.y_ctrl_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_cd, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                
                if self.banks_per_group == 4:
                    label_cd = f"BANK GROUP {'C' if i == 0 else 'D'}" if text else ""
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_cd, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+self.bank_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_cd, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+self.bank_height*2+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.y_ctrl_height-2*BUFFER_VIS], label="Y-CTRL" if text else "", facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+self.bank_height*2+self.y_ctrl_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_cd, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)
                    self.add_box(ax, [x_origin+lr*(self.mem_shoreline-self.bank_width)+BUFFER_VIS, y_origin+offset+i*bank_group_height+self.bank_height*3+self.y_ctrl_height+BUFFER_VIS], [self.bank_width-2*BUFFER_VIS, self.bank_height-2*BUFFER_VIS], label=label_cd, facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR)


        # x ctrl
        self.add_box(ax, [x_origin+self.bank_width+BUFFER_VIS, y_origin + BUFFER_VIS], [self.mem_shoreline-self.bank_width*2-2*BUFFER_VIS, 2*bank_group_height + -2*BUFFER_VIS], label="X-CTRL" if text else "", facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR, rotation=-90)
        self.add_box(ax, [x_origin+self.bank_width+BUFFER_VIS, y_origin+offset + BUFFER_VIS], [self.mem_shoreline-self.bank_width*2-2*BUFFER_VIS, 2*bank_group_height + -2*BUFFER_VIS], label="X-CTRL" if text else "", facecolor=MEMORY_COLOR, edgecolor=OUTLINE_COLOR, rotation=-90)

        # tsv region
        self.add_box(ax, [x_origin+BUFFER_VIS, y_origin+bank_group_height*2+BUFFER_VIS], [self.mem_shoreline-2*BUFFER_VIS, self.tsv_height -2*BUFFER_VIS], label="", facecolor=TSV_ARRAY_COLOR, edgecolor=OUTLINE_COLOR)
        
        # tsvs left side
        self.add_box(ax, [x_origin+2*BUFFER_VIS, y_origin+bank_group_height*2+2*BUFFER_VIS], [self.bank_width-3*BUFFER_VIS, self.tsv_height -4*BUFFER_VIS], label="DATA TSV" if text else "", facecolor=TSV_COLOR, edgecolor=OUTLINE_COLOR)
        
        # tsvs right side
        self.add_box(ax, [x_origin+self.mem_shoreline-self.bank_width+BUFFER_VIS, y_origin+bank_group_height*2+2*BUFFER_VIS], [self.bank_width-3*BUFFER_VIS, self.tsv_height -4*BUFFER_VIS], label="DATA TSV" if text else "", facecolor=TSV_COLOR, edgecolor=OUTLINE_COLOR)
        
        # tsvs center
        self.add_box(ax, [x_origin+self.bank_width+BUFFER_VIS, y_origin+bank_group_height*2+2*BUFFER_VIS], [self.mem_shoreline-2*self.bank_width-2*BUFFER_VIS, self.tsv_height -4*BUFFER_VIS], label="CTRL" if text else "", facecolor=TSV_COLOR, edgecolor=OUTLINE_COLOR, rotation=-90)

    def plot_compute_unit(self, ax, x_origin, y_origin, text=False):
        self.plot_compute_chiplet(ax, x_origin, y_origin)
        self.plot_hbm_co(ax, x_origin=x_origin+self.cu_width/2-self.mem_shoreline/2, y_origin=y_origin+self.core_height+self.mem_gap, text=text)
        self.plot_hbm_co(ax, x_origin=x_origin+self.cu_width/2-self.mem_shoreline/2, y_origin=y_origin-self.core_height-self.mem_gap-self.hbm_co_height, text=text)


