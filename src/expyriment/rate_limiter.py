class RateLimiter:
    def __init__(self, rate):
        self.rate = rate  # Allowed operations per second
        self.last_checked = None

    def is_allowed(self, current_time):
        if self.last_checked is None or current_time - self.last_checked >= 1000 / self.rate:
            self.last_checked = current_time
            return True
        return False
