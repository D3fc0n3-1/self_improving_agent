import abc
import json
import requests
import time
from typing import List, Dict, Any, Optional

# --- 1. LLM Backend Abstraction ---

class LLMBackend(abc.ABC):
    """
    Abstract Base Class for LLM backends.
    All concrete LLM implementations must adhere to this interface.
    """
    @abc.abstractmethod
    async def generate_content(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generates content based on a list of messages.
        Args:
            messages: A list of message dictionaries, e.g., [{"role": "user", "parts": [{"text": "Hello"}]}]
            kwargs: Additional parameters specific to the LLM API (e.g., temperature, max_tokens).
        Returns:
            The generated text content.
        """
        pass

    @abc.abstractmethod
    def get_model_name(self) -> str:
        """Returns the name of the LLM model used by this backend."""
        pass

class GeminiBackend(LLMBackend):
    """
    Concrete implementation for Google's Gemini API.
    Uses the provided fetch mechanism for API calls.
    """
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.api_key = "" # Canvas will provide this at runtime if empty

    async def generate_content(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generates content using the Gemini API.
        """
        print(f"--- Calling Gemini ({self.model_name}) ---")
        chat_history = messages
        payload = {
            "contents": chat_history,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_output_tokens", 2048),
            }
        }
        # Fixed the URL string formatting
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

        try:
            # Simulate fetch call in a synchronous environment for demonstration.
            # In a real browser/Node.js environment, this would be an actual fetch.
            # For this Python example, we'll use requests, assuming it's run in a context
            # where direct HTTP requests are allowed.
            # NOTE: In the actual Canvas environment, the `fetch` call is handled
            # by the system, and you would write it as shown in the prompt's
            # "Generating Text with LLMs via the Gemini API" section.
            # For a runnable Python script, we'll use `requests`.
            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status() # Raise an exception for HTTP errors
            result = response.json()

            if result.get("candidates") and len(result["candidates"]) > 0 and \
               result["candidates"][0].get("content") and \
               result["candidates"][0]["content"].get("parts") and \
               len(result["candidates"][0]["content"]["parts"]) > 0:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return text
            else:
                print(f"Gemini API response structure unexpected: {result}")
                return "Error: Could not generate content from Gemini API."
        except requests.exceptions.RequestException as e:
            print(f"Error calling Gemini API: {e}")
            return f"Error: Failed to connect to Gemini API: {e}"

    def get_model_name(self) -> str:
        return self.model_name

class OllamaBackend(LLMBackend):
    """
    Concrete implementation for Ollama.
    Assumes Ollama server is running locally (e.g., at http://localhost:11434).
    """
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.base_url = base_url

    async def generate_content(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generates content using the Ollama API.
        Ollama's /api/generate expects a 'prompt' or a 'messages' array for chat models.
        We'll convert the chat history to a single prompt for simplicity if needed,
        or use the 'messages' format if the model supports it directly.
        """
        print(f"--- Calling Ollama ({self.model_name}) ---")
        full_prompt = ""
        for msg in messages:
            role = msg["role"]
            # Ensure 'parts' key exists and is a list with at least one dictionary
            if "parts" in msg and isinstance(msg["parts"], list) and len(msg["parts"]) > 0 and isinstance(msg["parts"][0], dict) and "text" in msg["parts"][0]:
                text = msg["parts"][0]["text"]
                if role == "user":
                    full_prompt += f"\nUser: {text}"
                elif role == "model":
                    full_prompt += f"\nAssistant: {text}"
                elif role == "system":
                    full_prompt += f"\nSystem: {text}"
            else:
                print(f"Warning: Skipping message with unexpected structure: {msg}")

        full_prompt = full_prompt.strip()

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False, # We want the full response at once
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_output_tokens", 2048),
            }
        }

        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "Error: No response from Ollama.")
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}. Make sure Ollama server is running at {self.base_url}")
            return f"Error: Failed to connect to Ollama: {e}"

    def get_model_name(self) -> str:
        return self.model_name

# --- 2. Tool Abstraction ---

class Tool(abc.ABC):
    """
    Abstract Base Class for agent tools.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abc.abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Executes the tool's functionality."""
        pass

# Example Tools for a Programming Agent
class CodeInterpreterTool(Tool):
    def __init__(self):
        super().__init__("CodeInterpreter", "Executes Python code and returns the output or errors.")

    def execute(self, code: str) -> str:
        try:
            # In a real scenario, you'd want a sandboxed environment for code execution.
            # For demonstration, we'll use exec, but BE CAREFUL with untrusted code.
            # A more robust solution would involve a separate process or a dedicated
            # code execution service.
            print(f"\n--- Executing Code ---\n{code}\n----------------------")
            exec_globals = {}
            exec_locals = {}
            # Redirect stdout to capture print output
            import sys
            import io
            old_stdout = sys.stdout
            redirected_output = io.StringIO()
            sys.stdout = redirected_output

            try:
                exec(code, exec_globals, exec_locals)
                captured_output = redirected_output.getvalue()
                return f"Code executed successfully.\nOutput:\n{captured_output}"
            finally:
                sys.stdout = old_stdout # Restore stdout

        except Exception as e:
            return f"Error executing code: {e}"

class FileSystemTool(Tool):
    def __init__(self):
        super().__init__("FileSystem", "Reads from or writes to files.")

    def execute(self, action: str, path: str, content: Optional[str] = None) -> str:
        try:
            if action == "read":
                with open(path, 'r') as f:
                    return f.read()
            elif action == "write":
                with open(path, 'w') as f:
                    f.write(content)
                return f"Successfully wrote to {path}"
            else:
                return "Invalid file system action. Use 'read' or 'write'."
        except Exception as e:
            return f"Error with file system operation: {e}"

# Example Tool for a Server Administration Agent
class SSHClientTool(Tool):
    def __init__(self):
        super().__init__("SSHClient", "Executes commands on a remote server via SSH.")

    def execute(self, command: str, server_ip: str, username: str, password: str = None) -> str:
        # This is a placeholder. Real SSH interaction requires libraries like 'paramiko'.
        # For demonstration, we'll just simulate.
        print(f"\n--- Simulating SSH Command on {server_ip} as {username} ---\nCommand: {command}\n---------------------------------------------------------")
        if "ls" in command:
            return "Simulated SSH Output: file1.txt file2.log"
        elif "systemctl status" in command:
            return "Simulated SSH Output: service is active (running)"
        else:
            return f"Simulated SSH Output for '{command}': Command executed."

# --- 3. Agent Class ---

class Agent:
    """
    A generic self-improving AI agent.
    """
    def __init__(self,
                 name: str,
                 specialty: str,
                 llm_backend: LLMBackend,
                 tools: List[Tool],
                 initial_persona: str,
                 max_iterations: int = 5):
        self.name = name
        self.specialty = specialty
        self.llm = llm_backend
        self.tools = {tool.name: tool for tool in tools}
        self.persona = initial_persona
        self.memory: List[Dict[str, str]] = [] # Simple memory for chat history
        self.learned_lessons: List[str] = [] # For self-improvement
        self.max_iterations = max_iterations

        self._initialize_system_prompt()

    def _initialize_system_prompt(self):
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools.values()])
        self.system_prompt = f"""
        You are {self.name}, an expert {self.specialty}.
        Your goal is to assist the user with their requests related to {self.specialty}.
        You have access to the following tools:
        {tool_descriptions}

        To use a tool, respond with a JSON object in the format:
        {{
            "tool_name": "ToolName",
            "arguments": {{
                "arg1": "value1",
                "arg2": "value2"
            }}
        }}
        After using a tool, you will receive the tool's output.
        If you need to provide a final answer or ask for more information, respond with plain text.

        Current learned lessons (apply these to improve your performance):
        {chr(10).join(self.learned_lessons) if self.learned_lessons else "None yet."}

        {self.persona}
        """

    async def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_output_tokens: int = 2048) -> str:
        """Helper to call the LLM backend."""
        # Ensure messages are in the correct format for the LLM
        formatted_messages = []
        for msg in messages:
            # Assuming 'parts' should be a list of dicts, each with a 'text' key
            if 'parts' in msg and isinstance(msg['parts'], list):
                 formatted_messages.append(msg)
            elif 'text' in msg: # Handle simpler formats if needed, adapt to LLM requirements
                 formatted_messages.append({"role": msg.get("role", "user"), "parts": [{"text": msg["text"]}]})
            else:
                 print(f"Warning: Skipping message with unknown format: {msg}")

        return await self.llm.generate_content(formatted_messages, temperature=temperature, max_output_tokens=max_output_tokens)


    async def _reflect_and_learn(self, task_description: str, conversation_history: List[Dict[str, str]], outcome: str):
        """
        Agent reflects on its performance and learns lessons.
        This is a simplified self-improvement mechanism.
        """
        reflection_prompt = f"""
        You are {self.name}, an expert {self.specialty}.
        You have just completed a task.
        Task: {task_description}
        Conversation History:
        {json.dumps(conversation_history, indent=2)}
        Outcome: {outcome}

        Based on the task, conversation, and outcome, reflect on your performance.
        What went well? What could have been done better?
        Identify one specific, actionable lesson you can learn to improve your future performance for similar tasks.
        Format your lesson as a concise sentence. If no specific lesson is identified, just say "No new lesson."
        """
        # Prepare messages for reflection LLM call, ensuring correct format
        reflection_messages = [
            {"role": "system", "parts": [{"text": self.system_prompt}]},
            {"role": "user", "parts": [{"text": reflection_prompt}]}
        ]
        reflection_result = await self._call_llm(reflection_messages, temperature=0.5, max_output_tokens=200)

        if reflection_result and reflection_result.strip() != "No new lesson.":
            self.learned_lessons.append(reflection_result.strip())
            print(f"\n--- Agent {self.name} Learned Lesson ---")
            print(f"New Lesson: {reflection_result.strip()}")
            self._initialize_system_prompt() # Re-initialize system prompt with new lessons
            print("----------------------------------")
        else:
            print(f"\n--- Agent {self.name} Reflection ---")
            print("No new lesson identified.")
            print("----------------------------------")


    async def run(self, task_description: str) -> str:
        """
        Runs the agent's loop to complete a task.
        """
        print(f"\n--- Agent {self.name} (using {self.llm.get_model_name()}) starting task ---")
        print(f"Task: {task_description}")

        # Initial conversation history with system prompt and user task
        conversation_history = [
            {"role": "system", "parts": [{"text": self.system_prompt}]},
            {"role": "user", "parts": [{"text": task_description}]}
        ]

        final_output = ""
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1} ---")

            # Call LLM with the current conversation history
            llm_response = await self._call_llm(conversation_history)
            print(f"LLM Response: {llm_response}")

            try:
                # Attempt to parse the LLM response as JSON for a tool call
                response_json = json.loads(llm_response)
                tool_name = response_json.get("tool_name")
                tool_args = response_json.get("arguments", {})

                if tool_name and tool_name in self.tools:
                    tool = self.tools[tool_name]
                    print(f"--- Calling Tool: {tool_name} with args: {tool_args} ---")
                    tool_output = tool.execute(**tool_args)
                    print(f"Tool Output: {tool_output}")

                    # Append the LLM's tool call and the tool's output to history
                    conversation_history.append({"role": "model", "parts": [{"text": llm_response}]})
                    # For tool outputs, add a specific role or structure if the LLM backend supports it.
                    # Gemini expects "tool" role for function call responses. Ollama typically handles
                    # this by simply appending to the conversation. Adapting for a generic approach:
                    # Using a custom 'tool_output' role for internal tracking and potentially
                    # reformatting for the LLM in _call_llm if needed.
                    conversation_history.append({"role": "tool", "parts": [{"text": tool_output}]}) # Use 'tool' role as per some LLM conventions
                else:
                    print(f"Warning: LLM tried to call unknown tool '{tool_name}' or malformed tool call.")
                    # Append the LLM's response (even if it was a bad tool call attempt)
                    conversation_history.append({"role": "model", "parts": [{"text": llm_response}]})
                    # Append an error message as the "tool output"
                    conversation_history.append({"role": "tool", "parts": [{"text": "Error: Invalid tool call or tool not found."}]})
                    final_output = llm_response # If it tried to call a tool but failed, the response itself might be the final relevant output
                    break # Stop iteration if tool call failed or was invalid
            except json.JSONDecodeError:
                # If the response is not valid JSON, assume it's a final answer or plain text response
                final_output = llm_response
                conversation_history.append({"role": "model", "parts": [{"text": llm_response}]})
                break # Stop iteration as a non-JSON response is treated as final
            except Exception as e:
                print(f"Error during tool execution or parsing: {e}")
                final_output = f"An internal error occurred during tool execution: {e}"
                # Append the LLM's response and the error message to history
                conversation_history.append({"role": "model", "parts": [{"text": llm_response}]})
                conversation_history.append({"role": "tool", "parts": [{"text": f"Error: {e}"}]})
                break # Stop iteration on unexpected errors

            time.sleep(1) # Simulate some processing time before the next iteration

        print(f"\n--- Agent {self.name} Task Finished ---")
        print(f"Final Output: {final_output}")

        # Self-improvement step after the task is finished
        await self._reflect_and_learn(task_description, conversation_history, final_output)

        return final_output

# --- Main Execution ---

async def main():
    # Initialize LLM Backends
    # Ensure you have set up your Gemini API key or have Ollama running
    # gemini_backend = GeminiBackend(model_name="gemini-2.0-flash")
    # For testing without API key, use Ollama:
    ollama_backend = OllamaBackend(model_name="llama3") # Ensure llama3 is pulled in Ollama

    # --- Create Specialized Agents ---

    # 1. Programming Agent
    programming_tools = [CodeInterpreterTool(), FileSystemTool()]
    programming_persona = "You are a meticulous and efficient Python programmer. You prioritize clean, readable, and functional code. You always try to test your code if possible."
    programming_agent = Agent(
        name="CodeMaster",
        specialty="Programming and Coding",
        llm_backend=ollama_backend, # Using ollama for demonstration
        tools=programming_tools,
        initial_persona=programming_persona
    )

    # 2. Server Administration Agent
    server_admin_tools = [SSHClientTool()]
    server_admin_persona = "You are a highly reliable and security-conscious Linux server administrator. Your goal is to maintain system stability, optimize performance, and troubleshoot issues efficiently."
    server_admin_agent = Agent(
        name="SysGuard",
        specialty="Server Administration",
        llm_backend=ollama_backend, # Using ollama for demonstration
        tools=server_admin_tools,
        initial_persona=server_admin_persona
    )

    # --- Run Agent Tasks ---

    # Task for Programming Agent
    print("\n\n===== Running Programming Agent Task =====")
    # await programming_agent.run("Write a Python function that calculates the factorial of a number. Then, save it to a file named 'factorial.py'.")

    # Task for Server Administration Agent
    print("\n\n===== Running Server Administration Agent Task =====")
    # await server_admin_agent.run("Check the status of the 'nginx' service on server 192.168.1.100 using user 'admin' and password 'mypass'.")

    # Demonstrate self-improvement by running the programming agent task again
    # It should now incorporate the "learned lesson" from the previous run if any.
    print("\n\n===== Running Programming Agent Task (Second Time, with potential self-improvement) =====")
    # await programming_agent.run("Write a Python function that calculates the nth Fibonacci number. Include comments.")

    # Example task that uses the Code Interpreter Tool
    print("\n\n===== Running Programming Agent with Code Interpreter =====")
    await programming_agent.run("Use the CodeInterpreter tool to calculate 2 + 2 and print the result.")


# In a Jupyter notebook, you can directly await the main coroutine
# if get_ipython() is not None and 'IPKernelApp' in get_ipython().config:
# This check is not strictly necessary in modern Jupyter versions,
# direct await usually works.

# Removed the __main__ block with asyncio.run()
await main()

