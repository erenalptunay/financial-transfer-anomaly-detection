from functools import wraps
import time


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"⏱ Starting '{func.__name__}'...")
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"✅ Finished '{func.__name__}' in {elapsed:.2f} seconds.")
        return result
    return wrapper
