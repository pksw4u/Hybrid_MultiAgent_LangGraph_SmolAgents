class FeedbackMonitor:
    def __init__(self):
        self.performance_log = {}

    def evaluate(self, agent, sub_task):
        # Placeholder: always returns True (success)
        # Implement actual evaluation logic (correctness, latency, etc.)
        return True

    def log_performance(self, agent_name, sub_task, result, success):
        self.performance_log.setdefault(agent_name, []).append({
            'sub_task': sub_task,
            'result': result,
            'success': success
        })
