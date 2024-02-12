import os
import time

import numpy as np
import streamlit as st
from faiss import IndexFlatL2
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


st.set_page_config(
    page_title="Customer Service Bot",
    page_icon="❓",
    layout="wide",
)


def add_message(msg, agent="ai", stream=True):
    if stream and isinstance(msg, str):
        msg = stream_str(msg)

    with st.chat_message(agent):
        if stream:
            output = st.write_stream(msg)
        else:
            output = msg
            st.write(msg)

    st.session_state.messages.append(dict(agent=agent, content=output))


@st.cache_resource
def get_client():
    api_key = os.environ["MISTRAL_API_KEY"]
    return MistralClient(api_key=api_key)


CLIENT: MistralClient = get_client()


PROMPT = """
You are a customer service bot answering a query
from a user.

Here is the user information:

---------------------
{user_info}
---------------------

Here is additional context that may be relevant to
answer the user query:

---------------------
{context}
---------------------

Given the user info and the context, answer the following query.
If there is not enough information, decline to answer.
Do not output anything that can't be answered from the previous information.

Query: {query}
Answer:
"""


def reply(query: str, index: IndexFlatL2, **user_data):
    embedding = embed(query)
    embedding = np.array([embedding])

    _, indexes = index.search(embedding, k=3)
    context = [st.session_state.chunks[i] for i in indexes.tolist()[0]]

    user_info = "\n".join(
        f"{key}: {value}" for key, value in user_data.items()
    )

    messages = [
        ChatMessage(
            role="user",
            content=PROMPT.format(
                context=context, query=query, user_info=user_info
            ),
        )
    ]
    response = CLIENT.chat_stream(model="mistral-small", messages=messages)

    add_message(stream_response(response))


@st.cache_resource
def build_index():
    with open("faq.md") as fp:
        text = fp.read()

    chunks = [chunk.strip() for chunk in text.split("#") if chunk.strip()]

    st.sidebar.info(f"Indexing {len(chunks)} chunks.")
    progress = st.sidebar.progress(0)

    embeddings = []
    for i, chunk in enumerate(chunks):
        embeddings.append(embed(chunk))
        progress.progress((i + 1) / len(chunks))

    embeddings = np.array(embeddings)

    dimension = embeddings.shape[1]
    index = IndexFlatL2(dimension)
    index.add(embeddings)

    st.session_state.chunks = chunks
    return index


def stream_str(s, speed=250):
    for c in s:
        yield c
        time.sleep(1 / speed)


def stream_response(response):
    for r in response:
        yield r.choices[0].delta.content


@st.cache_data
def embed(text: str):
    return CLIENT.embeddings("mistral-embed", text).data[0].embedding


if st.sidebar.button("🔴 Reset conversation"):
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["agent"]):
        st.write(message["content"])


index = build_index()

name = st.sidebar.text_input("Username", "Neo")
plan = st.sidebar.selectbox("Plan", ["free", "premium", "enterprise"])
credits = st.sidebar.number_input(
    "Credits", 0.0, 1000.0, 100.0, format="%.2f"
)

query = st.chat_input("Ask something")

if not st.session_state.messages:
    add_message(
        """
This is a simple demonstration of how to use a large language model
and a vector database to implement a customer service bot that
personalized with user data.

This appplication uses [Mistral](https://mistral.ai) as language model,
so to deploy it you will need a corresponding API key.

Read the
[documentation](https://github.com/apiad/service-bot/blob/main/README.md)
or [browse the code](https://github.com/apiad/service-bot) in Github.""",
    )

    add_message(
        """In the sidebar you will find a set of user-defined values
        that simulate a user account.
        Change those values and ask something related to them to see the
        chatbot in action.
        """
    )

    add_message(
        """
        Ready to answer your questions. If you don't know where to start,
        just ask me to suggest you some questions.
            """
    )


if query:
    add_message(query, agent="human", stream=False)
    reply(query, index, name=name, plan=plan, credits=credits)

    if "balloon" in query:
        st.balloons()
