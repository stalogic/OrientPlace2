import atexit
import time
import signal

# 用于存储每个函数的统计信息
_function_stats = {}
# 用于存储整个程序的总耗时
_total_time = 0


class TimeTrackerMeta(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class TimeTracker(metaclass=TimeTrackerMeta):
    def __init__(self):
        self.total_time = 0

    def timeit(self, func):
        def timeit_wrapper(*args, **kwargs):
            global _function_stats
            func_name = func.__name__
            if func_name not in _function_stats:
                _function_stats[func_name] = {
                    "count": 0,
                    "total_time": 0,
                    "max_time": float("-inf"),
                    "min_time": float("inf"),
                }

            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time

            stats = _function_stats[func_name]
            stats["count"] += 1
            stats["total_time"] += elapsed
            if elapsed > stats["max_time"]:
                stats["max_time"] = elapsed
            if elapsed < stats["min_time"]:
                stats["min_time"] = elapsed

            self.total_time += elapsed
            print(f"函数 {func_name} 耗时: {elapsed:.4f} 秒")
            return result

        return timeit_wrapper

    def trackit(self, func):
        def trackit_wrapper(*args, **kwargs):
            global _function_stats
            func_name = func.__name__
            if func_name not in _function_stats:
                _function_stats[func_name] = {
                    "count": 0,
                    "total_time": 0,
                    "max_time": float("-inf"),
                    "min_time": float("inf"),
                }

            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time

            stats = _function_stats[func_name]
            stats["count"] += 1
            stats["total_time"] += elapsed
            if elapsed > stats["max_time"]:
                stats["max_time"] = elapsed
            if elapsed < stats["min_time"]:
                stats["min_time"] = elapsed

            self.total_time += elapsed
            return result

        return trackit_wrapper

    def print_total(self):
        global _total_time
        _total_time = self.total_time
        # print(f"程序总共消耗的时间: {_total_time:.4f} 秒")
        if len(_function_stats) > 0:
            print("每个函数的统计信息：")
        for func_name, stats in sorted(_function_stats.items(), key=lambda item: item[1]["total_time"]):
            if "count" not in stats or stats["count"] == 0:
                continue
            avg_time = stats["total_time"] / stats["count"]
            print(f"函数名: {func_name}")
            print(f"  调用次数: {stats['count']}")
            print(f"  累计耗时: {stats['total_time']:.4f} 秒")
            print(f"  最大耗时: {stats['max_time']:.4f} 秒")
            print(f"  平均耗时: {avg_time:.4f} 秒")
            print(f"  最小耗时: {stats['min_time']:.4f} 秒")
    
    def handle_signal(self, signum, frame):
        self.print_total()
        raise SystemExit(0)


# 注册程序结束时的操作
_tracker = TimeTracker()
atexit.register(_tracker.print_total)

# 暴露两个装饰器
timeit = _tracker.timeit
trackit = _tracker.trackit
