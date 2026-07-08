class State:
    def __init__(self, state_input):
        """
        Initialize State with either a string or dictionary.

        Args:
            state_input: Either 'steadystate' or a dict with transient parameters
        """
        if isinstance(state_input, str):
            if state_input.lower() == "steadystate":
                self.state_type = "steady_state"
                self.time_duration = 0.0
                self.time_step = 0.0
                self.scheme = None
            else:
                raise ValueError(f"Unknown state string: {state_input}")
        elif isinstance(state_input, dict):
            self.state_type = "transient"
            self.time_duration = state_input.get("time_duration")
            self.time_step = state_input.get("time_step")
            self.scheme = state_input.get("scheme")
        else:
            raise TypeError("state_input must be a string or dictionary")
