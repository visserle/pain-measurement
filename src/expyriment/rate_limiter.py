class RateLimiter:
    """
    Rate limiter class to limit the number of operations per second.
    """
    def __init__(self, rate):
        self.rate = rate  # Allowed operations per second
        self.last_checked = None

    def is_allowed(self, current_time):
        """
        Execute the operation if the rate limiter allows it.
        
        Expired time is calculated in milliseconds.
        """
        if self.last_checked is None or current_time - self.last_checked >= 1000 / self.rate:
            self.last_checked = current_time
            return True
        return False
