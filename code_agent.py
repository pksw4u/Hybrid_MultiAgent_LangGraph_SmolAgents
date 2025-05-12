from smolagents import CodeAgent as SmolCodeAgent, InferenceClientModel

class CodeAgent:
    def __init__(self):
        self.agent = SmolCodeAgent(
            tools=[],
            model=InferenceClientModel()
        )

    def run(self, task):
        for step in self.agent.run(task):
            yield f"[CodeAgent] {step}"
