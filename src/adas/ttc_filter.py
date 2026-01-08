class TTCFilter:
    def __init__(self, alpha=0.3, min_persist_frames=3):
        """
        alpha: EMA smoothing factor (0.2–0.4 works well)
        min_persist_frames: frames required before state escalation
        """
        self.alpha = alpha
        self.min_persist = min_persist_frames

        self.ttc_ema = None
        self.state_counter = {
            "CAUTION": 0,
            "WARNING": 0,
            "CRITICAL": 0,
        }

    def update(self, raw_ttc, raw_state):
        """
        Returns:
          smoothed_ttc, stable_state
        """

        # ---- EMA smoothing ----
        if raw_ttc is None:
            self.ttc_ema = None
        elif self.ttc_ema is None:
            self.ttc_ema = raw_ttc
        else:
            self.ttc_ema = self.alpha * raw_ttc + (1 - self.alpha) * self.ttc_ema

        # ---- Persistence gating ----
        stable_state = raw_state

        for s in self.state_counter:
            if raw_state == s:
                self.state_counter[s] += 1
            else:
                self.state_counter[s] = 0

        if raw_state in self.state_counter:
            if self.state_counter[raw_state] < self.min_persist:
                stable_state = "NORMAL"

        return self.ttc_ema, stable_state
