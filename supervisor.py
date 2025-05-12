import os
import ast
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from agents.web_search_agent import WebSearchAgent
from agents.youtube_agent import YouTubeAgent
from agents.image_agent import ImageAgent
from agents.math_agent import MathAgent
from agents.code_agent import CodeAgent
from feedback import FeedbackMonitor

class SupervisorState(BaseModel):
    input: str
    messages: List[Any] = Field(default_factory=list)
    pending_tasks: List[str] = Field(default_factory=list)
    current_task: str = Field(default="")
    agent_type: str = Field(default="")
    results: Dict[str, List[str]] = Field(default_factory=dict)
    feedback_needed: bool = Field(default=False)
    output: str = Field(default="")
    current_task_output: List[str] = Field(default_factory=list)
    class Config:
        arbitrary_types_allowed = True

from langchain_google_genai import ChatGoogleGenerativeAI
from feedback import FeedbackMonitor
from agents.web_search_agent import WebSearchAgent
from agents.youtube_agent import YouTubeAgent
class SupervisorAgent:
    def __init__(self):
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
            print("LLM initialized.")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            self.llm = None
        self.feedback_monitor = FeedbackMonitor()
        self.agents = {
            'web_search': WebSearchAgent(),
            'youtube': YouTubeAgent(),
            'image': ImageAgent(),
            'math': MathAgent(),
            'code': CodeAgent(),
        }
        self.graph = self._build_graph()
        print("LangGraph built.")

    def _decompose_task_node(self, state: SupervisorState) -> SupervisorState:
        print("--- Entering decompose_task node ---")
        if self.llm is None:
            print("LLM not available. Cannot decompose task.")
            return state.copy(update={"output": "Error: AI model not available for task decomposition."})
        prompt = f"Decompose the following question into a list of distinct sub-tasks. Respond ONLY as a Python list of strings, e.g., ['sub-task 1', 'sub-task 2']. Question: {state.input}"
        try:
            sub_tasks_str = self.llm.invoke(prompt).content

            # --- Strip code block markers if present ---
            s = sub_tasks_str.strip()
            if s.startswith("```"):
                lines = s.splitlines()
                # Remove opening triple backticks and optional language
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[0].strip().lower() in ("python", "py"):
                    lines = lines[1:]
                # Remove closing triple backticks if present
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                sub_tasks_str = "\n".join(lines).strip()
            else:
                sub_tasks_str = s

            sub_tasks = ast.literal_eval(sub_tasks_str)
            if not isinstance(sub_tasks, list) or not all(isinstance(task, str) for task in sub_tasks):
                print(f"Warning: LLM did not return a valid list of strings. Output: {sub_tasks_str}")
                sub_tasks = [state.input]
            print(f"Decomposed into tasks: {sub_tasks}")
            return state.copy(update={"pending_tasks": sub_tasks, "messages": state.messages + [AIMessage(content=f"Decomposed into: {sub_tasks}")]})
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing LLM sub-tasks output: {e}. Output string: {sub_tasks_str}")
            return state.copy(update={"pending_tasks": [state.input], "messages": state.messages + [AIMessage(content=f"Error decomposing task. Proceeding with original input.")]})
        except Exception as e:
            print(f"Error during task decomposition: {e}")
            return state.copy(update={"output": f"Error during task decomposition: {e}", "pending_tasks": []})

    def _route_task_node(self, state: SupervisorState) -> SupervisorState:
        print("--- Entering route_task node ---")
        if not state.pending_tasks:
            print("No pending tasks. Routing to format_final_answer.")
            return state
        current_task = state.pending_tasks[0]
        remaining_tasks = state.pending_tasks[1:]
        agent_type = self._route_task_helper(current_task)
        print(f"Routing task '{current_task}' to agent: {agent_type}")
        return state.copy(update={
            "current_task": current_task,
            "agent_type": agent_type,
            "pending_tasks": remaining_tasks,
            "messages": state.messages + [AIMessage(content=f"Assigned task '{current_task}' to {agent_type} agent.")]
        })

    def _make_agent_node(self, agent):
        def node(state: SupervisorState) -> SupervisorState:
            print(f"--- Entering agent node for {agent.__class__.__name__} ---")
            task = state.current_task
            agent_output = []
            try:
                for thought in agent.run(task):
                    agent_output.append(thought)
                updated_results = state.results.copy()
                updated_results[task] = agent_output
                return state.copy(update={
                    "results": updated_results,
                    "current_task_output": agent_output,
                    "messages": state.messages + [AIMessage(content=f"Result for '{task}':\n" + "\n".join(agent_output))]
                })
            except Exception as e:
                print(f"Error executing agent for task '{task}': {e}")
                error_output = [f"[{agent.__class__.__name__}] Error executing task '{task}': {e}"]
                updated_results = state.results.copy()
                updated_results[task] = error_output
                return state.copy(update={
                    "results": updated_results,
                    "current_task_output": error_output,
                    "messages": state.messages + [AIMessage(content=f"Error for '{task}': {e}")]
                })
        node.__name__ = agent.__class__.__name__.lower() + "_node"
        return node

    def _evaluate_feedback_node(self, state: SupervisorState) -> SupervisorState:
        print("--- Entering evaluate_feedback node ---")
        task = state.current_task
        agent_output = state.current_task_output
        if not agent_output:
            print("No agent output to evaluate feedback on.")
            return state.copy(update={"feedback_needed": False})
        feedback_needed = not self.feedback_monitor.evaluate(agent_output, task)
        print(f"Feedback needed for task '{task}': {feedback_needed}")
        return state.copy(update={"feedback_needed": feedback_needed})

    def _process_next_step(self, state: SupervisorState) -> str:
        print("--- Entering process_next_step node ---")
        if state.feedback_needed:
            print("Decision: Feedback needed. Routing to feedback_handler.")
            return "handle_feedback"
        elif state.pending_tasks:
            print("Decision: More pending tasks. Routing to route_task.")
            return "continue_processing"
        else:
            print("Decision: No pending tasks. Routing to format_final_answer.")
            return "final_answer"

    def _handle_feedback_node(self, state: SupervisorState) -> SupervisorState:
        print("--- Entering handle_feedback node ---")
        task = state.current_task
        print(f"Handling feedback for task: {task}")
        updated_pending_tasks = state.pending_tasks + [task + " (needs review)"]
        print(f"Marking task '{task}' for review and adding back to pending tasks.")
        return state.copy(update={"feedback_needed": False, "pending_tasks": updated_pending_tasks,
                                  "messages": state.messages + [AIMessage(content=f"Task '{task}' needs review.")]})

    def _format_final_answer_node(self, state: SupervisorState) -> SupervisorState:
        print("--- Entering format_final_answer node ---")
        final_output_parts = ["Final Answer:"]
        if not state.results:
            final_output_parts.append("No results were generated.")
        else:
            for task, output in state.results.items():
                final_output_parts.append(f"\nResult for '{task}':")
                final_output_parts.extend(output)
        final_output = "\n".join(final_output_parts)
        print("--- Exiting format_final_answer node ---")
        return state.copy(update={"output": final_output, "messages": state.messages + [AIMessage(content=final_output)]})

    def _route_task_helper(self, sub_task: str) -> str:
        print(f"Routing helper for task: {sub_task}")
        if "search" in sub_task.lower() or "find information" in sub_task.lower():
            return 'web_search'
        if "video" in sub_task.lower() or "youtube" in sub_task.lower():
            return 'youtube'
        if "image" in sub_task.lower() or "picture" in sub_task.lower():
            return 'image'
        if any(word in sub_task.lower() for word in ["calculate", "math", "solve", "equation"]):
            return 'math'
        if any(word in sub_task.lower() for word in ["code", "python", "execute", "script"]):
            return 'code'
        return 'web_search'

    def _build_graph(self):
        graph = StateGraph(SupervisorState)
        graph.add_node("decompose_task", self._decompose_task_node)
        graph.add_node("route_task", self._route_task_node)
        for name, agent in self.agents.items():
            graph.add_node(name, self._make_agent_node(agent))
        graph.add_node("evaluate_feedback", self._evaluate_feedback_node)
        graph.add_node("process_next_step", self._process_next_step)
        graph.add_node("handle_feedback", self._handle_feedback_node)
        graph.add_node("format_final_answer", self._format_final_answer_node)
        graph.set_entry_point("decompose_task")
        graph.add_edge("decompose_task", "route_task")
        def select_agent_node(state: SupervisorState) -> str:
            return state.agent_type
        graph.add_conditional_edges(
            "route_task",
            select_agent_node,
            {name: name for name in self.agents.keys()}
        )
        for name in self.agents.keys():
            graph.add_edge(name, "evaluate_feedback")
        graph.add_conditional_edges(
            "evaluate_feedback",
            self._process_next_step,
            {
                "handle_feedback": "handle_feedback",
                "continue_processing": "route_task",
                "final_answer": "format_final_answer"
            }
        )
        graph.add_edge("handle_feedback", "route_task")
        graph.set_finish_point("format_final_answer")
        try:
            compiled_graph = graph.compile()
            print("LangGraph compiled successfully.")
            return compiled_graph
        except Exception as e:
            print(f"Error compiling LangGraph: {e}")
            return None

    def handle_user_message(self, message: str, history: List[Any]):
        print(f"\n--- Handling User Message: {message} ---")
        if self.graph is None:
            print("Graph is not compiled. Cannot process message.")
            yield "[Supervisor] Error: Agent system is not initialized correctly."
            return
        initial_state = SupervisorState(input=message, messages=history + [HumanMessage(content=message)])
        try:
            for step in self.graph.stream(initial_state):
                for node_name, state_update in step.items():
                    print(f"--- Node: {node_name} ---")
                    if 'output' in state_update and state_update['output']:
                        yield f"[Supervisor] Final Output Update: {state_update['output']}"
                    elif 'current_task' in state_update and state_update['current_task']:
                        yield f"[Supervisor] Processing Task: {state_update['current_task']}"
                    elif 'current_task_output' in state_update and state_update['current_task_output']:
                        yield f"[Supervisor] Task Result: {' '.join(state_update['current_task_output'])[:100]}..."
                    elif 'feedback_needed' in state_update:
                        yield f"[Supervisor] Feedback Needed: {state_update['feedback_needed']}"
        except Exception as e:
            print(f"\nAn error occurred during graph execution: {e}")
            yield f"[Supervisor] An error occurred: {e}"

