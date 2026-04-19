"""
Usage examples for pi-agent — Anthropic, OpenAI, and Vertex providers.
"""

import asyncio
import base64
from pi_agent import (
    Agent, AgentOptions, AgentTool, AgentToolResult,
    ModelConfig, TextContent, ImageContent, UserMessage,
    ContextWindowBreakdown, ContextBucket,
    AgentEvent, AbortController,
)


# ---------------------------------------------------------------------------
# Example tool
# ---------------------------------------------------------------------------

async def bash_execute(tool_call_id, params, signal, on_update) -> AgentToolResult:
    import subprocess
    try:
        result = subprocess.run(
            params["command"], shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr or "(no output)"
    except subprocess.TimeoutExpired:
        output = "Command timed out"
    return AgentToolResult(content=[TextContent(text=output)])


bash_tool = AgentTool(
    name="bash",
    label="Bash",
    description="Run a shell command. Returns stdout and stderr.",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
    execute=bash_execute,
)


# ---------------------------------------------------------------------------
# Event subscriber that prints streaming text + context window summary
# ---------------------------------------------------------------------------

def make_subscriber(agent: Agent):
    def on_event(event: AgentEvent, signal: AbortController):
        if event.type == "message_update":
            for block in event.message.content:
                if block.type == "text":
                    print(block.text, end="", flush=True)
                elif block.type == "thinking":
                    print(f"[thinking] {block.thinking[:80]}...", end="", flush=True)

        elif event.type == "tool_execution_start":
            print(f"\n\n[{event.tool_name}] {event.args}")

        elif event.type == "tool_execution_end":
            status = "error" if event.is_error else "ok"
            print(f"[{event.tool_name} → {status}]")

        elif event.type == "agent_end":
            usage = agent.state.context_usage
            if usage:
                print(f"\n\n{usage.summary()}")

    return on_event


# ---------------------------------------------------------------------------
# Anthropic example
# ---------------------------------------------------------------------------

async def run_anthropic():
    agent = Agent(AgentOptions(
        model_config=ModelConfig(
            model="claude-sonnet-4-5",
            provider="anthropic",
            api_key="YOUR_ANTHROPIC_API_KEY",
            context_window=200_000,
        ),
        initial_system_prompt="You are a helpful coding assistant.",
        initial_tools=[bash_tool],
    ))

    # Manually set context window breakdown (populated by higher-level session layer)
    # In production, this comes from the system prompt builder after it knows token counts.
    def on_event(event, signal):
        if event.type == "agent_end":
            usage = agent.state.context_usage
            if usage and not usage.breakdown:
                # Example: populate breakdown from known sizes
                usage.breakdown = ContextWindowBreakdown(
                    system_prompt=ContextBucket(tokens=189),
                    system_tools=ContextBucket(tokens=6800, count=5),
                    memory_files=ContextBucket(tokens=53, count=1),
                )
            if usage:
                print(f"\n{usage.summary()}")

    agent.subscribe(on_event)
    await agent.prompt("List the files in the current directory.")
    await agent.wait_for_idle()


# ---------------------------------------------------------------------------
# OpenAI example
# ---------------------------------------------------------------------------

async def run_openai():
    agent = Agent(AgentOptions(
        model_config=ModelConfig(
            model="gpt-4o",
            provider="openai",
            api_key="YOUR_OPENAI_API_KEY",
            context_window=128_000,
            supports_images=True,
        ),
        initial_system_prompt="You are a helpful assistant.",
        initial_tools=[bash_tool],
    ))
    agent.subscribe(make_subscriber(agent))
    await agent.prompt("What is 2 + 2? Show your work in a bash command.")
    await agent.wait_for_idle()


# ---------------------------------------------------------------------------
# OpenAI example with an image attached to the prompt
# ---------------------------------------------------------------------------

async def run_openai_with_image():
    # Load image as base64 (replace with real image path)
    image_path = "screenshot.png"
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        image = ImageContent(data=image_data, mime_type="image/png")
    except FileNotFoundError:
        print("No image file found, skipping image example")
        return

    agent = Agent(AgentOptions(
        model_config=ModelConfig(
            model="gpt-4o",
            provider="openai",
            api_key="YOUR_OPENAI_API_KEY",
            supports_images=True,
        ),
    ))
    agent.subscribe(make_subscriber(agent))

    # Attach image to prompt
    await agent.prompt(UserMessage(content=[
        TextContent(text="Describe what you see in this image."),
        image,
    ]))
    await agent.wait_for_idle()


# ---------------------------------------------------------------------------
# Vertex AI example
# ---------------------------------------------------------------------------

async def run_vertex():
    agent = Agent(AgentOptions(
        model_config=ModelConfig(
            model="gemini-2.0-flash-001",
            provider="vertex",
            # Either api_key or vertex_project + vertex_location
            vertex_project="my-gcp-project",
            vertex_location="us-central1",
            context_window=1_000_000,
            supports_thinking=True,
        ),
        initial_system_prompt="You are a helpful assistant.",
        initial_tools=[bash_tool],
        thinking_level="medium",
    ))
    agent.subscribe(make_subscriber(agent))
    await agent.prompt("Explain async/await in Python in two sentences.")
    await agent.wait_for_idle()


# ---------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

async def run_openrouter():
    agent = Agent(AgentOptions(
        model_config=ModelConfig(
            model="anthropic/claude-sonnet-4-5",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key="YOUR_OPENROUTER_API_KEY",
            headers={"HTTP-Referer": "https://myapp.com"},
            context_window=200_000,
        ),
        initial_system_prompt="You are helpful.",
    ))
    agent.subscribe(make_subscriber(agent))
    await agent.prompt("Hello!")
    await agent.wait_for_idle()


if __name__ == "__main__":
    # Pick one to run:
    asyncio.run(run_anthropic())
    # asyncio.run(run_openai())
    # asyncio.run(run_vertex())
    # asyncio.run(run_openrouter())
