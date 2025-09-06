import streamlit as st
import openai
from langfuse import observe
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Langfuse Test Chat", page_icon="💬")


@observe()
def call_chatgpt(messages):
    """Call OpenAI ChatGPT API with Langfuse instrumentation"""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, max_tokens=500, temperature=0.7
    )

    return response.choices[0].message.content


def main():
    st.title("💬 Langfuse Test Chat")
    st.caption("A minimal ChatGPT wrapper with Langfuse instrumentation")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to chat about?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Call ChatGPT with Langfuse instrumentation
                    response = call_chatgpt(st.session_state.messages)
                    st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # Sidebar with info
    with st.sidebar:
        st.header("Configuration")
        st.info(
            "Make sure to set your environment variables:\n\n"
            "- OPENAI_API_KEY\n"
            "- LANGFUSE_PUBLIC_KEY\n"
            "- LANGFUSE_SECRET_KEY\n"
            "- LANGFUSE_HOST (optional)"
        )

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
