import cProfile
import linecache
import os
import tracemalloc


# https://docs.python.org/3/library/tracemalloc.html
class MemoryProfiler:
    def __enter__(self):
        tracemalloc.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        self.display_top(snapshot)

    @staticmethod
    def display_top(snapshot, key_type='lineno', limit=3):
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))
        top_stats = snapshot.statistics(key_type)

        print(f"Top {limit} lines")
        for index, stat in enumerate(top_stats[:limit], 1):
            frame = stat.traceback[0]
            # replace "/path/to/module/file.py" with "module/file.py"
            filename = os.sep.join(frame.filename.split(os.sep)[-2:])
            print(f"#{index}: {filename}:{frame.lineno}: "
                  f"{stat.size / 1024:.1f}.1f KiB")
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print(f'    {line}')

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            print(f"{len(other)} other: {size / 1024:.1f} KiB")
        total = sum(stat.size for stat in top_stats)
        print(f"Total allocated size: {total / 1024:.1f} KiB")


class LineProfiler:
    def __init__(self, *stats_args, **stats_kwargs):
        self.pr = cProfile.Profile()
        self.stats_args = stats_args
        self.stats_kwargs = stats_kwargs

    def __enter__(self):
        self.pr.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pr.disable()
        self.pr.print_stats(*self.stats_args, **self.stats_kwargs)
