class DependencyTracker:
    def __init__(self):
        self.active = False
        self._deps: set[str] = set()

    def start(self):
        self._deps = set()
        self.active = True

    def stop(self) -> set[str]:
        self.active = False
        return self._deps

    def register(self, var_name: str):
        if self.active:
            self._deps.add(var_name)
