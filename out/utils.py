class PerfForHardware:
    def __init__(self, device_name, perf, size=None):
        self.device = device_name
        self.perf = perf
        self.size = size
        pass
    
class Plotter:
    def __init__(self, params_setup: Conv1dParams, *perf_data, log_scale=False):
        self.labels = ["TORCH", "ACPP", "DPCPP"]
        self.colors = [
            "#e66349",
            "#E63946",
            "#0071C5",
        ]  # Custom colors for each category
        self.log_scale = log_scale

        self.params = params_setup
        self.perf_data = perf_data
        pass

    def plot(self, ax=None):
        hardwares = [run.device for run in self.perf_data[0]]  # Assume all lists are aligned
        x = np.arange(len(hardwares))  # X-axis positions
        width = 0.25  # Bar width

        if ax is None:
            fig, ax = plt.subplots()

        if self.log_scale:
            f = np.log
        else:
            f = lambda x: x

        # Iterate over each dataset (TORCH, ACPP, DPCPP)
        for i, (data, label, color) in enumerate(zip(self.perf_data, self.labels, self.colors)):
            values = [f(run.perf) if run.perf is not None else np.nan for run in data]
            ax.bar(x + (i - 1) * width, values, width, label=label, color=color)

        # Labels and formatting
        ax.set_ylabel(f'{"log-Performance" if self.log_scale else "Performance"}')
        ax.set_title(self.params.title())
        ax.set_xticks(x)
        ax.set_xticklabels(hardwares)
        ax.legend()
        ax.grid()