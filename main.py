import gradio as gr
from supervisor import SupervisorAgent

def chat_fn(message, history):
    supervisor = SupervisorAgent()
    yield from supervisor.handle_user_message(message, history)

def main():
    gr.ChatInterface(
        fn=chat_fn,
        title="Hybrid Multi-Agent AI Assistant",
        description="Ask GAIA Level 1 benchmark questions. Intermediate reasoning and final answers will be streamed in real time."
    ).launch()

if __name__ == "__main__":
    main()
