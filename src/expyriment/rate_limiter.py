class RateLimiter:
    """
    Rate limiter class to limit the number of operations per second.

    With an optional interval-based limiting feature to enforce operation execution at precise, predefined intervals.
    This feature allows for more granular control over the timing of operations, making it ideal for scenarios
    where it's crucial to maintain a consistent rate of operation execution over time. When enabled, this mode
    ensures operations are initiated only at specific, regular intervals (e.g., every 100 ms), as opposed to
    simply limiting the number of operations per second without regard to their temporal distribution.
    """

    def __init__(self, rate, use_intervals=False):
        """
        Initialize the rate limiter with the specified rate and interval-based limiting option.

        - rate is the allowed number of operations per second.
        - use_intervals is a flag to enable interval-based limiting.
        """
        self.rate = rate  # Allowed operations per second
        self.use_intervals = use_intervals  # Whether to use interval-based limiting
        if use_intervals:
            # Interval in milliseconds for allowed operations
            self.interval = 1000 / rate
            self.next_allowed_time = 0  # Track the next allowed time directly
        else:
            self.last_checked = (
                None  # Track the last checked time for simple rate limiting
            )

    def reset(self):
        """
        Reset the rate limiter to allow immediate operation.
        """
        if self.use_intervals:
            self.next_allowed_time = 0
        else:
            self.last_checked = None

    def is_allowed(self, current_time):
        """
        Check if the operation is allowed at the current time, optionally considering specific intervals.

        - current_time is expected to be in milliseconds.
        """
        if self.use_intervals:
            if current_time >= self.next_allowed_time:
                self.next_allowed_time = (
                    current_time + self.interval - (current_time % self.interval)
                )
                return True
            return False
        else:
            if (
                self.last_checked is None
                or current_time - self.last_checked >= 1000 / self.rate
            ):
                self.last_checked = current_time
                return True
            return False
