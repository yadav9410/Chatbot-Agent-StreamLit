
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

st.set_page_config(page_title='My AI Chat', layout='centered')

st.title("🤖 The Groq Chatbot")
st.write("A fully integrated, memory-enabled AI assistant.")


# The Sidebar
with st.sidebar:

    st.header("⚙️ Configuration")

    user_api_key = st.text_input(
        'Enter the Groq API Key:',
        type='password'
    )

    st.info('Your key is required to wake up the AI Brain')

    # ---------------- NEW ----------------
    # System Prompt
    persona = st.text_area(
        "System Prompt:",
        value="You are a helpful assistant."
    )

    # ---------------- NEW ----------------
    # Reset Chat & Apply Persona
    if st.button("Reset Chat & Apply Persona"):

        # Clear previous conversation
        st.session_state.messages = []

        # Add system prompt as FIRST message
        st.session_state.messages.append(
            SystemMessage(content=persona)
        )

        st.success("New persona applied!")
        st.rerun()


# ------ Memory Vault -------

if 'messages' not in st.session_state:
    st.session_state.messages = []


# ----- Display History ---------

for msg in st.session_state.messages:

    # Don't display system prompt
    if isinstance(msg, SystemMessage):
        continue

    # LangChain message objects
    if hasattr(msg, "type"):

        if msg.type == "human":
            role = "user"

        elif msg.type == "ai":
            role = "assistant"

        else:
            continue

        with st.chat_message(role):
            st.markdown(msg.content)


# ----- Chat Input and Logic -----

if user_query := st.chat_input('Say something to the AI...'):

    if not user_api_key:

        st.error('Please enter your api key in the sidebar first!')

    else:

        # Display user message
        with st.chat_message('user'):
            st.markdown(user_query)

        # Store user message
        from langchain_core.messages import HumanMessage

        st.session_state.messages.append(
            HumanMessage(content=user_query)
        )


        # Initialise the Brain

        llm = ChatGroq(
            model='openai/gpt-oss-20b',
            temperature=0.7,
            api_key=user_api_key
        )


        with st.spinner('AI is thinking....'):

            # The FIRST message is SystemMessage
            response = llm.invoke(
                st.session_state.messages
            )

            bot_answer = response.content


        # Store assistant response

        from langchain_core.messages import AIMessage

        st.session_state.messages.append(
            AIMessage(content=bot_answer)
        )


        # Display assistant response

        with st.chat_message('assistant'):
            st.markdown(bot_answer)
