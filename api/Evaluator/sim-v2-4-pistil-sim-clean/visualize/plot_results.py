

class PlotResultsBase:
    def __init__(self, results_dir):
        self.title_size = 12
        self.axtitle_size = 10
        self.xlabel_size = 10
        self.ylabel_size = 10
        self.xtick_size = 8
        self.ytick_size = 8
        self.legend_size = 10
        
        self.marker_size = 6
        self.marker_edge_width = .8

        self.large_marker_size = 8
        self.small_marker_size = 6

        self.dpi = 100
        
        self.results_dir = results_dir
        
        #self.my_color_map = ["#A51C30", "#3872B2", "#EC8F9C", "#FF6600", "#ECDD7B", "#808080"]
        self.my_color_map = ["#ACC3B1", "#B7D1E2", "#F1A151", "#F1D365", "#DAD7D0", "#668C87"]
        self.my_color_map_extended = ["#AED994", "#71CAB1", "#63CDC0", "#02A6AF", "#94C9DB", "#78ADD1", "#859FB8", "#9FADB8", "#E2EDF0"]
        self.my_markers = ["d", "*", "v", "o",]

        