import time

def sleep_to_next_interval(interval=10):
    """Ждёт N секунд до следующего запроса"""
    delta = interval - (time.time() % interval) + 1
    print(f'sleep {delta:.2f}s')
    time.sleep(delta)
