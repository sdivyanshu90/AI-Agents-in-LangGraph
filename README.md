# AI Agents in LangGraph

This repository contains my personal notes, code, and projects from the **[AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)** course by **[DeepLearning.AI](https://www.deeplearning.ai/)**, offered in collaboration with **[LangChain](https://www.langchain.com/)** and **[Tavily](https://www.tavily.com/)**.

## Introduction

This course provides a comprehensive, hands-on journey into building modern, sophisticated AI agents. We begin by constructing a foundational agent from scratch using Python and a Large Language Model (LLM), establishing a deep understanding of the core components: the LLM's reasoning capabilities and the essential Python code that enables action and interaction with the outside world.

From there, we transition to LangGraph, a powerful library from the LangChain ecosystem. LangGraph allows us to rebuild our agent within a stateful, graph-based framework. This is a pivotal shift, as it enables the creation of cyclical, controllable, and robust applications that can manage complex, multi-step logic—a necessity for any serious agentic system.

A key focus of the course is enhancing agent capabilities with Agentic Search. We explore how specialized tools, like Tavily, move beyond traditional search (which returns a list of links) to provide multiple, synthesized answers in a structured, agent-friendly format. This pre-processed, high-quality data dramatically improves an agent's ability to reason and respond effectively.

Finally, we delve into production-ready features, including:

* Persistence: Enabling agents to manage state across multiple sessions, switch between conversations, and reload previous states.

* Human-in-the-Loop: Designing systems where agents can pause to request human clarification or approval, ensuring safety and quality.

* Streaming: Providing a responsive user experience by streaming agent thoughts and actions in real-time.

The course culminates in building a practical "Essay Writer" agent, a multi-step workflow that replicates the process of a human researcher, tying together every concept we've learned.

## Course Topics

This repository is organized around the key modules of the course. Below is a detailed breakdown of the concepts covered in each section.

<details>
<summary><strong>1. Build an Agent from Scratch</strong></summary>

Before diving into sophisticated frameworks, this foundational module guides us through the process of building an agent from first principles. The primary goal is to demystify the "magic" of AI agents by clearly delineating the division of labor between the Large Language Model (LLM) and the Python code that surrounds it. This hands-on approach reveals that an agent is not a monolithic black box, but a symbiotic system of (1) a reasoning brain and (2) a deterministic body of code.

The "Brain": The Role of the LLM

The LLM (e.g., GPT-4, Claude 3) serves as the agent's central reasoning engine. Its job is not to do things in the real world, but to think and plan. We learn that the "prompt" is far more than just a user's question; it is the agent's "operating system." A well-designed agent prompt typically includes:

System Message: A detailed charter that defines the agent's persona, its capabilities, its constraints, and the high-level goal it should pursue.

Tool Definitions: A structured (often in JSON schema) list of all the functions (tools) the agent is allowed to use. This is the agent's "toolbox."

Conversation History / Scratchpad: A log of previous interactions (user, AI, and tool outputs) that provides the agent with memory and context.

The LLM's core task is to process this entire prompt and, based on the user's latest query, decide on one of two possible outputs:

A Final Answer: If it has sufficient information, it generates a direct response to the user.

A Tool Call: If it lacks information, it outputs a structured request to call one of its defined tools (e.g., {"tool": "search", "arguments": {"query": "latest AI news"}}).

The "Body": The Role of the Python Code

The Python code is the "executor" or the "chassis" of the agent. It handles all the deterministic tasks that the LLM cannot. Its responsibilities form the core control loop of the agent, often referred to as a ReAct (Reason + Act) loop.

This loop, which we build manually, looks like this:

* Receive Input: Get a query from the user.

* Format Prompt: Add the user's query to the conversation history.

* Invoke LLM: Send the complete prompt (system message, tools, history) to the LLM.

* Parse Output: Receive the LLM's response.

* Decision Point (The Core Logic):

If the output is a Final Answer: The loop is complete. Send the answer to the user.

If the output is a Tool Call: The loop continues. The Python code must:
a.  Parse the call: Identify which tool to use (e.g., search_tool) and what arguments to pass (e.g., "latest AI news").
b.  Execute the tool: This is the key step. The Python code—not the LLM—actually calls the underlying function (e.g., def search_tool(query): ...).
c.  Get the Result: The tool function returns its output (e.g., a list of search results).
d.  Update History: The Python code appends this tool output to the conversation history (e.g., as a ToolMessage).
e.  Go to Step 3: The entire updated history (now including the tool call and its result) is sent back to the LLM for the next reasoning step.

The "Why": Understanding the Fundamentals

By building this loop from scratch, we gain a deep appreciation for what agents truly are. We learn that the LLM is a non-deterministic planner, while our code provides the deterministic scaffolding that connects that plan to the real world (via APIs, file systems, etc.).

This manual approach also highlights the limitations that lead directly to frameworks like LangGraph. Managing the state (the conversation history) is manual. Implementing complex, branching logic (e.g., "if search results are empty, try a different query, otherwise summarize") becomes a messy series of if/else statements. Handling errors, retries, and interruptions is brittle. This module perfectly sets the stage by showing us both the power of the agent concept and the necessity of a more robust framework to manage its complexity.

</details>

<details>
<summary><strong>2. LangGraph Components</strong></summary>

After building an agent from scratch, we graduate to LangGraph. This module is a paradigm shift. Instead of thinking of an agent as a single, linear ReAct loop, LangGraph encourages us to model agentic workflows as a stateful graph. This is a far more powerful and flexible abstraction, allowing us to build applications with cycles, branches, and sophisticated control flow.

A LangGraph application is built from three core components:

1. The State (The "Memory")

At the heart of every LangGraph application is a central state object. This is typically a Python TypedDict (or a Pydantic model) that defines the "memory" of our graph. As the graph executes, this single state object is passed from node to node. Each node can read from the state and add or update information in it.

A simple state might look like this:
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages] # A list of all messages in the chat
    current_files: list[str]
    final_result: Optional[str]
```

The state object is the single source of truth for the entire application. It's how memory is managed, how tools pass results back to the agent, and how the graph tracks its own progress.

2. Nodes (The "Do-ers")

Nodes are the "compute steps" of the graph. Each node is simply a Python function or callable that receives the current state as input and returns a partial state object (a dictionary) with its updates. LangGraph then merges this partial update into the main state.

There are two primary types of nodes:

* Compute Nodes: These do the work.

* agent_node: A function that calls the LLM with the current state.messages to get the next step (either a tool call or a final answer).

* tool_node: A function that parses the latest message, executes the requested tool (e.g., search), and returns the tool's output.

* utility_node: Any other Python function, e.g., one that formats data, validates input, or logs metrics.

Special Nodes:

* ENTRYPOINT: The node where the graph begins execution.

* END: A special node that signifies the graph has finished its work.

3. Edges (The "Deciders")

Edges connect the nodes and define the flow of logic. This is where LangGraph's power truly lies.

Standard Edges: graph.add_edge("A", "B"). This is a simple, directional link. After node "A" finishes, always run node "B" next.

Conditional Edges: graph.add_conditional_edges(...). This is the most important component. A conditional edge routes the application to different nodes based on the current state.

This works by first routing to a special "router" node (which is just a Python function). This function inspects the state and returns a string indicating the name of the next node to visit.

A common pattern is:

* An agent_node runs, adding an AIMessage to the state.

* A conditional edge routes to a router_function.

The router_function inspects the new AIMessage:

If the message contains a tool_call, it returns the string "call_tools".

If the message does not contain a tool_call (i.e., it's a final answer), it returns the string "__end__".

The graph, based on this string, then routes to either the tool_node (named "call_tools") or the END node.

The Graph Object (The "Executor")

You define your state, your nodes, and your edges, and add them to a StateGraph object. Finally, you call graph.compile(). This method takes your abstract definition and builds a concrete, runnable LangGraph object. This compiled graph exposes the same familiar .invoke(), .stream(), and .batch() methods as any other LangChain object (LCEL).

By modeling our agent as a graph, we gain:

* Modularity: Each node is a discrete, testable unit of logic.

* Control: Complex, branching logic is no longer a nested if/else mess; it's a clean, visualizable set of conditional edges.

* Cyclicality: We can easily create loops (e.g., tool_node -> agent_node -> tool_node), which are the foundation of all ReAct agents.

* Statefulness: The state is managed for us by the graph, making it trivial to build applications with persistent memory.

LangGraph shifts the developer's focus from writing imperative control-flow code to declaratively designing the agent's state and logic flow.

</details>

<details>
<summary><strong>3. Agentic Search Tools</strong></summary>

This module addresses a fundamental bottleneck for AI agents: accessing external information. While agents are good at reasoning, their knowledge is limited to their training data. To be useful, they must be able to search the real-time web. This module explores the critical difference between traditional search and agentic search.

The Problem with Traditional Search for Agents

A traditional search engine (like Google) is designed for humans. When a human searches, they get back a list of 10 blue links and text snippets. This is a perfect format for a human, who can use their intuition to:

Scan the snippets and titles.

Click a promising link.

Read and synthesize the content of the webpage.

Evaluate its trustworthiness.

Click "back," try another link, and integrate the new information.

This workflow is terrible for an LLM. An LLM cannot "click a link." The old, brittle workflow for an agent was:

* Agent: Decides to search. Calls Google Search("latest EU AI regulation").

* Tool: Returns 10 links and snippets.

* Agent: Receives the 10 links. It must now guess which link is best.

* Agent: Decides to scrape_website("some-news-link.com/article").

* Tool: Scrapes the URL, returning a wall of messy, ad-filled HTML.

* Agent: Receives the messy HTML. It must now try to clean and summarize this text, which may or may not contain the answer.

* Agent: If the answer isn't there, it must repeat steps 4-6 for the next link.

This process is incredibly slow, unreliable (scrapers break constantly), expensive (many LLM calls), and often pollutes the agent's context window with irrelevant data.

The "Agentic Search" Solution (e.g., Tavily)

"Agentic Search" is a new category of tool, highlighted in this course, that is built for agents, not humans. As the course description notes, it "retrieves multiple answers in a predictable format."

It acts as a meta-service. Instead of returning links, an agentic search tool performs the entire human workflow on its own, optimized backend:

Agent: Decides to search. Calls tavily_search("latest EU AI regulation").

Tavily Service:
a.  Receives the query.
b.  Performs a large-scale traditional web search.
c.  Identifies dozens of promising links.
d.  Crawls and reads the content of those pages in parallel.
e.  Synthesizes and cross-references the information from all sources.
f.  Finds the most relevant facts, figures, and summaries.
g.  Prepares a structured, agent-friendly response.

Tool: Returns a clean, structured JSON object to the agent.

A response from an agentic search tool doesn't look like links. It looks like answers. For example:
```json
[
  {
    "content": "The EU AI Act was officially adopted in May 2024. It follows a risk-based approach, categorizing AI systems into 'unacceptable', 'high', 'limited', and 'minimal' risk tiers.",
    "source_url": "euparliament.europa.eu/news/...",
    "score": 0.95
  },
  {
    "content": "Enforcement of the act will be staggered, with full implementation expected by 2026. Fines for non-compliance can be as high as 7% of a company's global annual revenue.",
    "source_url": "[techcrunch.com/eu-ai-act-details/](https://techcrunch.com/eu-ai-act-details/)...",
    "score": 0.92
  }
]
```

Integrating with LangGraph

This new tool becomes a simple, powerful node in our graph.

The agent_node decides to search.

The conditional edge routes to the tavily_search_node.

This node calls the Tavily API and gets the structured JSON response.

It adds this clean, information-dense content to the state.messages as a ToolMessage.

The graph routes back to the agent_node.

The LLM now has high-quality, pre-digested, citable information already in its context. It doesn't need to scrape, clean, or guess. It can move directly to the final step of synthesizing these facts into a high-quality answer for the user.

This approach is a "force multiplier" for agents. It dramatically improves speed, reliability, and the quality of the final output by offloading the brittle, time-consuming work of data extraction to a specialized service, allowing the LLM to focus on what it does best: reasoning.

</details>

<details>
<summary><strong>4. Persistence and Streaming</strong></summary>

This module focuses on transforming our agent from a "demo script" into a "production-ready application." Persistence and Streaming are two sides of the same coin: they are critical for building robust, user-friendly systems that can handle real-world interactions.

Part 1: Persistence (The "Long-Term Memory")

Persistence is the ability to save and load the state of our agent. In the context of LangGraph, this means saving the state object. Without persistence, every time a user refreshes their browser or the server restarts, the agent's memory is wiped clean. This is unacceptable for any real application.

This course highlights persistence as the key to:

* Long-Running Tasks: An agent writing an essay (as in the capstone project) might take several minutes and dozens of steps. If the connection drops, persistence allows the agent to resume exactly where it left off.

* Conversation History: The most obvious use case. A user expects to close a chat window and return hours or days later, with the agent remembering their entire conversation.

* Conversation Switching: As noted in the course, a user might be working with an agent on "Project A," then need to quickly ask about "Project B." A persistent system can save the state for "Project A," load the state for "Project B," and then seamlessly switch back.

* Reloading Previous States: This enables powerful "undo" functionality or version history. A user could ask, "What was the essay draft like three steps ago?" With persistence, the agent can retrieve and display that historical state.

How it's Implemented in LangGraph:

* LangGraph has a built-in Checkpointing system designed specifically for persistence.

* Define a Backend: You choose where to save the state. LangGraph provides checkpointers for Sqlite, Postgres, Redis, or you can create a custom one (e.g., for Firebase, DynamoDB).

* Configure the Graph: You attach the checkpointer to your LangGraph object when you compile it.

* Identify Conversations: You must provide a unique ID for each conversation (e.g., a thread_id or conversation_id).

Once configured, the checkpointer automatically saves the entire state object to your database after every single step (or at configurable intervals). When you invoke or stream the graph, you provide the conversation_id. The graph, before executing any logic, will first check the database for a saved state for that ID. If it finds one, it loads it into memory before running the first node. This entire process is seamless and automatic, giving our agent a robust, long-term memory.

Part 2: Streaming (The "Real-Time Feedback")

Streaming is the key to a good User Experience (UX). An agent that "thinks" for 30 seconds with a blank screen or a spinning wheel will be abandoned by users, who will assume it's broken.

Streaming is the ability to send back partial results as they are generated. Instead of waiting for the final answer, the user gets a real-time feed of the agent's "thoughts" and "actions."

This provides two key benefits:

* Responsiveness: The user gets immediate feedback, which feels fast and interactive.

* Transparency: The user can see what the agent is doing, which builds trust.

Instead of a 30-second wait, a streaming UI would show:

"Okay, I need to research the EU AI Act."

"Calling search tool... (Tool Call: tavily_search)"

"I've found two key articles. Reading them now..."

"Drafting the summary..."

"Here is the final answer..." (Final Answer)

How it's Implemented in LangGraph:

The compiled LangGraph object has a .stream() method (in addition to .invoke()).

`.invoke()` runs the entire graph and returns only the final state.

`.stream()` runs the graph step-by-step and yields the partial state updates as they happen.

On a web server (like FastAPI), you can use a StreamingResponse to pass these yielded chunks directly to the frontend. The frontend JavaScript can then listen to this stream and update the UI in real-time, displaying each thought, tool call, and tool result as it's generated.

Conclusion:

Persistence gives the agent a past (long-term memory). Streaming gives the agent a present (real-time responsiveness). Together, these two features are non-negotiable for building professional, interactive, and trustworthy AI applications.

</details>

<details>
<summary><strong>5. Human in the loop</strong></summary>

This module addresses one of the most critical aspects of building safe and reliable agents: Human-in-the-Loop (HITL). We must acknowledge that agents are not (and may not be for some time) fully autonomous. They are powerful tools that assist humans. HITL is the practice of designing systems where an agent must pause its execution to request human input, clarification, or approval before proceeding.

This is not just a feature; it's an essential component for safety, quality control, and usability.

Why is HITL Essential?

To Resolve Ambiguity: The user's request is often vague.

User: "Write an essay on that topic we discussed yesterday."

Agent (without HITL): (Guesses) "Here is an essay on The Roman Republic."

Agent (with HITL): (Pauses) "Do you mean 'AI in Healthcare' or 'The Roman Republic'?"

To Ensure Safety: The agent is about to perform a high-stakes, irreversible, or destructive action.

User: "Delete all the files in the staging/ directory."

Agent (with HITL): (Pauses) "This will permanently delete 47 files. Are you absolutely, 100% sure? Please type 'CONFIRM' to proceed."

To Control Quality: The agent has completed a critical sub-task but is not fully confident in the result, or the task is subjective.

Agent (with HITL): (Pauses) "I have drafted the three-point outline for your essay. Does this structure look correct to you before I begin writing the full text?"

To Manage Cost: The agent is about to begin a complex, multi-step research loop that will consume significant time and API tokens.

Agent (with HITL): (Pauses) "This research request will involve analyzing approximately 50 documents and will cost an estimated $0.75 in API credits. Shall I proceed?"

Implementing HITL in LangGraph

LangGraph's stateful, graph-based model makes implementing HITL a natural and elegant pattern, not a "hack." It's just another node and edge in our graph.

Here is the common architecture:

The "Wait for Human" Node: We define a special node (e.g., human_input_node). This node's job is not to compute. Its job is to interrupt the graph's execution. LangGraph provides a pause_before mechanism or allows for explicit InterruptibleGraph configurations. When the graph hits this node, it pauses and saves its state (using the Persistence/Checkpointing from the previous module).

The Conditional Edge: How does the graph get to this node? A conditional edge.

An agent_node runs. It makes a decision.

A router_node inspects the agent's decision (which is now in the state).

If the decision is "request_clarification" or "request_approval", the router returns the string "wait_for_human".

If the decision is "call_tool", it returns "call_tools".

The graph's conditional edge then routes execution to the appropriate node.

The "Resuming" Flow:

The graph is now "paused" at the human_input_node.

The application (e.g., the web frontend) sees the agent's question ("Please approve this outline...").

The user provides their feedback ("Looks great, proceed!").

This new human message is sent back to our application.

Our application resumes the graph, passing the new human message as the input to the paused step.

The human_input_node receives this input, adds it to the state.messages, and the graph continues executing from where it left off (e.g., routing back to the agent_node for the next step).

HITL as a Spectrum

This module teaches us to think of HITL in several forms:

* Active (Approval): The agent must stop and wait (e.g., approving an outline).

* Passive (Editing): The agent produces a full draft and then presents it. The user can edit the draft. The edited draft is then fed back into the graph for a "revision" step. The human's edit becomes a "node" in the loop.

* Corrective (Feedback): The agent provides a final answer. The user gives it a "thumbs down." This feedback (the query + the bad answer) is logged and used to fine-tune the agent's prompts or logic later (offline HITL).

By building HITL directly into our agent's workflow, we create a collaborative tool that combines the speed and scale of an LLM with the judgment, expertise, and safety of a human user.

</details>

<details>
<summary><strong>6. Essay Writer (Capstone Project)</strong></summary>

The "Essay Writer" is the capstone project of the course, synthesizing every concept from the previous modules into a single, complex, and practical application. This project is not about building a simple "text-generation" tool; it's about replicating the workflow of a human researcher.

A human doesn't just sit down and write a 2,000-word essay from scratch. They follow a deliberate, multi-step, iterative process. Our goal is to model this exact process as a graph in LangGraph.

Deconstructing the "Researcher Workflow"

A good research workflow, which we model in our graph, looks like this:

Define and Clarify Topic: Understand the core question. (Requires Human-in-the-Loop).

Brainstorm & Outline: Create a logical structure for the argument. (Requires Agent Node).

Get Outline Approval: (Optional but recommended) Verify the structure with the user. (Requires Human-in-the-Loop).

Research & Gather Sources: For each point in the outline, find relevant information. (Requires Agentic Search Tools).

Read & Synthesize: Read the gathered sources and extract the key facts, quotes, and data relevant to each outline point. (Requires Agent Node).

Draft Section by Section: Write the text for each section individually, using only the synthesized research for that section. (Requires Agent Node, possibly a "Writer" specialist).

Review & Edit: Combine the sections and read the full draft for flow, grammar, and argument coherence. (Requires Agent Node, "Editor" specialist).

Format & Finalize: Add citations and a conclusion.

Building the "Essay Writer" Graph Architecture

This workflow is far too complex for a single agent. Instead, we build a multi-node graph where each step is a dedicated node or sub-graph.

The State: The StateGraph for this project is our "workbench" and is much more complex.
```python
class EssayWorkflowState(TypedDict):
    topic: str
    outline: Optional[str]
    research_data: Optional[dict[str, list[dict]]] # Maps outline_point -> list[search_results]
    draft_by_section: Optional[dict[str, str]]   # Maps outline_point -> "drafted text..."
    final_essay: Optional[str]
    messages: Annotated[list, add_messages]
```

The Nodes and Edges (The Workflow):

ENTRYPOINT -> clarify_topic_node (Agent):

This node takes the user's initial topic. It decides if the topic is clear enough to proceed or if it needs to ask for clarification.

Conditional Edge: Routes to human_input_node if ambiguous, or to create_outline_node if clear.

human_input_node (HITL):

This node is for all human interventions. It pauses the graph (using Persistence).

It waits for the user's reply (e.g., the topic clarification, or outline approval).

When the human replies, their message is added to state.messages and the graph resumes, typically routing back to the previous agent node to re-process.

create_outline_node (Agent):

Takes the state.topic, generates a detailed outline, and saves it to state.outline.

Edge: Routes to human_input_node to request outline approval.

research_node (Tool-Using Agent):

This node is a loop. It iterates through each point in the approved state.outline.

For each point, it calls the Agentic Search Tool (e.g., Tavily) to gather information.

It aggregates all this information, structured by outline point, into state.research_data.

The Streaming from this node is critical, showing the user: "Researching section 1...", "Researching section 2..."

drafting_node (Agent):

This node is the "Writer." It iterates through each key in state.research_data.

It writes the text for each section based on the outline point and its associated research.

It saves its work to state.draft_by_section.

edit_and_compile_node (Agent):

This node is the "Editor." It takes all the individual drafts from state.draft_by_section.

It combines them into a single, cohesive document.

It performs a final pass for flow, grammar, and adds an introduction/conclusion.

It saves the final text to state.final_essay.

edit_and_compile_node -> __end__:

The graph finishes. The final state, containing the completed essay, is returned to the user.

This capstone project is the perfect demonstration of LangGraph's power. It shows how to break a highly complex, valuable task into a manageable, observable, and controllable graph. It moves far beyond simple chat, creating a robust tool that collaborates with a user to produce a high-quality result. It is the perfect synthesis of all six course topics.

</details>

## Acknowledgement

This repository is for educational and learning purposes only. It contains my personal notes and implementations based on the **[AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)** course.

All course materials, concepts, and original code are the intellectual property of **[DeepLearning.AI](https://www.deeplearning.ai/)**, **[LangChain](https://www.langchain.com/)**, and **[Tavily](https://www.tavily.com/)**. I extend my sincere gratitude to them for creating this high-quality, practical, and insightful educational content.

Please refer to the official **[DeepLearning.AI](https://www.deeplearning.ai/)** and **[LangChain](https://www.langchain.com/)** websites for the original course.