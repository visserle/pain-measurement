class RateLimiter:
    """
    Rate limiter to limit the number of operations per second.

    Initializes with a specified rate and an optional interval-based limiting feature,
    to enforce operation execution at precise, predefined intervals.

    Parameters:
    - rate (int): allowed number of operations per second
    - use_intervals (bool, optional): enables interval-based limiting (default: False)
    """

    def __init__(
        self,
        rate: int,
        use_intervals: bool = False,
    ):
        self.rate = rate
        self.use_intervals = use_intervals
        if use_intervals:
            # Interval in milliseconds for allowed operations
            self.interval = 1000 / rate
            self.next_allowed_time = 0
        else:
            self.last_checked = None

    def reset(self) -> None:
        """
        Reset the rate limiter to allow immediate operation.
        """
        if self.use_intervals:
            self.next_allowed_time = 0
        else:
            self.last_checked = None

    def is_allowed(self, current_time_ms: int) -> bool:
        """
        Check if the operation is allowed at the current time, optionally considering
        specific intervals.

        - current_time_ms is expected to be in milliseconds.
        """
        if self.use_intervals:
            if current_time_ms >= self.next_allowed_time:
                self.next_allowed_time = (
                    current_time_ms + self.interval - (current_time_ms % self.interval)
                )
                return True
            return False
        else:
            if (
                self.last_checked is None
                or current_time_ms - self.last_checked >= 1000 / self.rate
            ):
                self.last_checked = current_time_ms
                return True
            return False
