import matplotlib.pyplot as plt

rc_stack = []

def rc_push(rc):
    rc_stack.append({**plt.rcParams})
    print("rc_push", len(rc_stack))
    plt.rcParams.update(rc)

def rc_pop():
    print("rc_pop", len(rc_stack))
    plt.rcParams.update(rc_stack.pop())

class rc_block:
    def __init__(self, rc): self.rc=rc
    def __enter__(self): rc_push(self.rc)
    def __exit__(self, exc_type, exc_val, exc_tb): rc_pop()

def rc_sizes(sm, md, lg, fig_sz):
    return {
        "font.size": sm, # controls default text sizes
        "axes.titlesize": lg, # fontsize of the axes title
        "axes.labelsize": md, # fontsize of the x and y labels
        "xtick.labelsize": sm, # fontsize of the tick labels
        "ytick.labelsize": sm, # fontsize of the tick labels
        "legend.fontsize": sm, # legend fontsize
        "figure.titlesize": lg, # fontsize of the figure title
        "figure.figsize": fig_sz, # figure size
    }

def create_plot_context(global_sizes, reload_functions):
    import os
    class MyPlot:
        def __init__(self, pdf_filename, sizes=None, figsize=None):
            if sizes is None: sizes = {**global_sizes}
            if figsize is not None: sizes["figure.figsize"] = figsize
            self.sizes = sizes
            self.pdf_filename = pdf_filename
            self.rc_block = None
        def __enter__(self):
            reload_functions()
            self.rc_block = rc_block(self.sizes)
            self.rc_block.__enter__()
        def __exit__(self, exc_type, exc_val, exc_tb):
            basename = os.path.splitext(self.pdf_filename)[0]
            if exc_type is None:
                plt.savefig(basename+'.png', bbox_inches='tight', dpi=300)
                plt.savefig(basename+'.pdf', bbox_inches='tight')
            plt.show()
            plt.close()
            if self.rc_block is not None: self.rc_block.__exit__(exc_type, exc_val, exc_tb)
    return MyPlot
