%%writefile app.py

import streamlit as st

from langchain_groq import ChatGroq

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

from langchain_community.tools import DuckDuckGoSearchRun

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multi-Agent AI",
    layout="centered"
)

st.title("🤖 Multi-Agent AI Assistant")

st.write(
    "Supervisor + Researcher + Calculator + Greeter"
)


# ============================================================
# 2. SIDEBAR - API KEY
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    user_api_key = st.text_input(
        "Enter the Groq API Key:",
        type="password"
    )

    st.info(
        "Enter your Groq API key to activate the agents."
    )


# ============================================================
# 3. MEMORY VAULT
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# 4. DISPLAY CHAT HISTORY
# ============================================================

for msg in st.session_state.messages:

    if isinstance(msg, HumanMessage):

        with st.chat_message("user"):
            st.markdown(msg.content)

    elif isinstance(msg, AIMessage):

        with st.chat_message("assistant"):
            st.markdown(msg.content)


# ============================================================
# 5. CREATE MULTI-AGENT SYSTEM
# ============================================================

def create_multi_agent(api_key):


    # ========================================================
    # BASE LLM
    # ========================================================

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key
    )


    # ========================================================
    # AGENT 1: WEB SEARCH TOOL
    # ========================================================

    search = DuckDuckGoSearchRun()


    def web_search(query: str) -> str:
        """
        Search the web for latest or factual information.
        """

        print("🔎 web_search called")

        return search.run(query)


    # ========================================================
    # AGENT 2: CALCULATOR TOOLS
    # ========================================================

    def add(a: float, b: float) -> float:
        """
        Adds two numbers.
        """

        print("➕ add called")

        return a + b


    def multiply(a: float, b: float) -> float:
        """
        Multiplies two numbers.
        """

        print("✖️ multiply called")

        return a * b


    def divide(a: float, b: float) -> float:
        """
        Divides two numbers.
        """

        print("➗ divide called")

        return a / b


    # ========================================================
    # SPECIALIST LLMs
    # ========================================================

    researcher_llm = llm.bind_tools(
        [web_search]
    )


    calculator_llm = llm.bind_tools(
        [add, multiply, divide]
    )


    # ========================================================
    # LANGGRAPH STATE
    # ========================================================

    class State(TypedDict):

        messages: Annotated[
            list,
            add_messages
        ]

        next: str

        steps: int


    # ========================================================
    # SUPERVISOR NODE
    # ========================================================

    def supervisor_node(state: State):

        steps = state.get(
            "steps",
            0
        )


        # Safety limit

        if steps >= 6:

            print("🛑 Step limit reached")

            return {
                "next": "end",
                "steps": steps + 1
            }


        # Conversation history

        history = state["messages"]


        # Supervisor prompt

        prompt = f"""
You are a supervisor managing three specialist agents.

researcher - can search the web.
Use for facts, prices, news, current data.

calculator - can add, multiply, divide.
Use for any arithmetic.

greeter - responds warmly to greetings and casual conversation.

Read the conversation so far and decide what must happen NEXT.

Rules:

- If the user is greeting you or making casual conversation,
  reply: greeter

- If a fact is still missing,
  reply: researcher

- If a number still needs to be computed,
  reply: calculator

- If the user's full request has already been answered,
  reply: end

Reply with exactly ONE word:

greeter
researcher
calculator
end

Conversation:

{history}
"""


        # Ask supervisor

        decision = (
            llm.invoke(prompt)
            .content
            .strip()
            .lower()
        )


        print(
            f"🧠 Supervisor decided: {decision}"
        )


        # Decide next agent

        if decision == "researcher":

            next_agent = "researcher"


        elif decision == "calculator":

            next_agent = "calculator"


        elif decision == "greeter":

            next_agent = "greeter"


        else:

            next_agent = "end"


        return {

            "next": next_agent,

            "steps": steps + 1

        }


    # ========================================================
    # RESEARCHER NODE
    # ========================================================

    def researcher_node(state: State):

        system = SystemMessage(
            content="""
You are the researcher.

You can ONLY search the web.

Do not do arithmetic.

Find the requested fact and state it clearly.

Another agent will handle mathematical calculations.
"""
        )


        response = researcher_llm.invoke(
            [system] + state["messages"]
        )


        return {
            "messages": [response]
        }


    # ========================================================
    # CALCULATOR NODE
    # ========================================================

    def calculator_node(state: State):

        system = SystemMessage(
            content="""
You are the calculator.

You can ONLY perform arithmetic using your tools.

Do not search the web.

Use numbers already present in the conversation.

Always call a calculator tool.

Never calculate mentally.
"""
        )


        response = calculator_llm.invoke(
            [system] + state["messages"]
        )


        return {
            "messages": [response]
        }


    # ========================================================
    # GREETER NODE
    # ========================================================

    def greeter_node(state: State):

        system = SystemMessage(
            content="""
You are a warm and friendly greeter.

Respond naturally and warmly to greetings
and casual conversation.

Keep the response short and friendly.

Do not use any tools.
"""
        )


        response = llm.invoke(
            [system] + state["messages"]
        )


        return {
            "messages": [response]
        }


    # ========================================================
    # ROUTE AFTER SUPERVISOR
    # ========================================================

    def route_after_supervisor(
        state: State
    ) -> Literal[
        "researcher",
        "calculator",
        "greeter",
        END
    ]:


        print(
            "📍 Route:",
            state["next"]
        )


        if state["next"] == "researcher":

            return "researcher"


        elif state["next"] == "calculator":

            return "calculator"


        elif state["next"] == "greeter":

            return "greeter"


        return END


    # ========================================================
    # TOOL NODES
    # ========================================================

    researcher_tool_node = ToolNode(
        [web_search]
    )


    calculator_tool_node = ToolNode(
        [
            add,
            multiply,
            divide
        ]
    )


    # ========================================================
    # RESEARCHER LOOP
    # ========================================================

    def researcher_should_continue(
        state: State
    ):


        last_message = state["messages"][-1]


        if last_message.tool_calls:

            return "researcher_tools"


        return "supervisor"


    # ========================================================
    # CALCULATOR LOOP
    # ========================================================

    def calculator_should_continue(
        state: State
    ):


        last_message = state["messages"][-1]


        if last_message.tool_calls:

            return "calculator_tools"


        return "supervisor"


    # ========================================================
    # BUILD LANGGRAPH
    # ========================================================

    builder = StateGraph(State)


    # --------------------------------------------------------
    # Add Nodes
    # --------------------------------------------------------

    builder.add_node(
        "supervisor",
        supervisor_node
    )


    builder.add_node(
        "researcher",
        researcher_node
    )


    builder.add_node(
        "calculator",
        calculator_node
    )


    builder.add_node(
        "greeter",
        greeter_node
    )


    builder.add_node(
        "researcher_tools",
        researcher_tool_node
    )


    builder.add_node(
        "calculator_tools",
        calculator_tool_node
    )


    # --------------------------------------------------------
    # START → SUPERVISOR
    # --------------------------------------------------------

    builder.add_edge(
        START,
        "supervisor"
    )


    # --------------------------------------------------------
    # SUPERVISOR → SPECIALIST
    # --------------------------------------------------------

    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor
    )


    # --------------------------------------------------------
    # RESEARCHER LOOP
    # --------------------------------------------------------

    builder.add_conditional_edges(
        "researcher",
        researcher_should_continue
    )


    builder.add_edge(
        "researcher_tools",
        "researcher"
    )


    # --------------------------------------------------------
    # CALCULATOR LOOP
    # --------------------------------------------------------

    builder.add_conditional_edges(
        "calculator",
        calculator_should_continue
    )


    builder.add_edge(
        "calculator_tools",
        "calculator"
    )


    # --------------------------------------------------------
    # GREETER → SUPERVISOR
    # --------------------------------------------------------

    builder.add_edge(
        "greeter",
        "supervisor"
    )


    # ========================================================
    # COMPILE GRAPH
    # ========================================================

    multi_agent = builder.compile()


    return multi_agent


# ============================================================
# 6. CHAT INPUT
# ============================================================

if user_query := st.chat_input(
    "Ask something..."
):


    # ========================================================
    # CHECK API KEY
    # ========================================================

    if not user_api_key:

        st.error(
            "Please enter your Groq API key in the sidebar."
        )

        st.stop()


    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    with st.chat_message("user"):

        st.markdown(user_query)


    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    user_message = HumanMessage(
        content=user_query
    )


    st.session_state.messages.append(
        user_message
    )


    # ========================================================
    # RUN MULTI-AGENT
    # ========================================================

    with st.spinner(
        "🤖 Agents are working..."
    ):


        try:


            # Create graph using
            # Streamlit API key

            multi_agent = create_multi_agent(
                user_api_key
            )


            # Initial LangGraph state

            initial_state = {

                "messages": st.session_state.messages,

                "steps": 0

            }


            # Run graph

            final_state = multi_agent.invoke(
                initial_state
            )


            # Get final answer

            bot_answer = (
                final_state["messages"][-1].content
            )


        except Exception as e:

            st.error(
                f"Agent error: {e}"
            )

            st.stop()


    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    assistant_message = AIMessage(
        content=bot_answer
    )


    st.session_state.messages.append(
        assistant_message
    )


    # ========================================================
    # DISPLAY ASSISTANT MESSAGE
    # ========================================================

    with st.chat_message("assistant"):

        st.markdown(bot_answer)
