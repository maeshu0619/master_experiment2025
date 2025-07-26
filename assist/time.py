import time

class TimeTracker:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """計測開始"""
        self.start_time = time.time()
        print("計測開始")

    def stop(self):
        """計測終了"""
        self.end_time = time.time()
        print("計測終了")

    def elapsed(self):
        """経過時間を返す"""
        if self.start_time is None or self.end_time is None:
            raise ValueError("計測が開始されていません。start()を呼び出してください。")
        return self.end_time - self.start_time
    
    def print_elapsed(self):
        """経過時間を表示"""
        print(f"経過時間: {self.end_time:.2f} - {self.start_time:.2f}秒")