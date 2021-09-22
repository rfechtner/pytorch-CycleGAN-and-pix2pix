from ray.tune.logger import TBXLoggerCallback

class CustomLoggerCallback(TBXLoggerCallback):
    def __init__(self):
        super(CustomLoggerCallback, self).__init__()

    def log_figure(self, tag, fig, step):
        self._file_writer.add_figure(
            tag, fig, global_step=step)
        self._file_writer.flush()