from smolagents import CodeAgent, InferenceClientModel

class MathAgent:
    def __init__(self):
        self.agent = CodeAgent(
            tools=[],
            model=InferenceClientModel()
        )

    def run(self, task):
        for step in self.agent.run(task):
            yield f"[MathAgent] {step}"
