# Customer Service Bot

> A simple streamlit app showcasing how to build a customer service bot to help users of a hypothetical service

## Check the app here

<iframe width="100%" height="600px" src="https://service-bot-demo.streamlit.app/"></iframe>

## Running

- Get an API key from [mistral.ai](https://mistral.ai).
- Create a virtualenv.
- Install `requirements.txt`.
- Create a file `.streamlit/secrets.toml` with a `MISTRAL_API_KEY="<your-mistral-key>"`.
- Run `streamlit run app.py`.

## Basic architecture

The appplication is a single-script [streamlit](https://streamlit.io) app.
It uses streamlit's chat UI elements to simulate a conversation with a chatbot that
can answer queries about to a hypothetical user in a service, mixing both
user specific data and general knowledge from a FAQ.

### Context

This app simulates a hypothetical service that contains a FAQ guide (in [`faq.md`](faq.md)).
To answer a specific user question, we combine a subset of relevant answers from the
FAQ with custom user data: a username, a plan, and a credit score.

### Question answering

The model `mistral-small` is employed to answer questions.

The user query is first embedded with the same `mistral-embed` model and queried against
a `faiss` index, where the closet matching chunk is extracted.

Afterward, a custom prompt is constructed that contains the user query, the custom user info,
and the retrieved chunks of text from the FAQ, and fed to the LLM. The response is streamed back
to the application.

### Chat management

The application relies on `st.chat_message` and `st.chat_input` as the main UI elements.

To stream text and simulate a typing behavior, instead of the classic `st.write`,
all text is displayed with `st.write_stream`, that receives an iterable of text chunks.

The text received from the LLM already comes in an iterable, so only a simple wraping
is necessary to obtain each text fragment.
However, the custom text that is sometimes displayed by the chatbot (like the hello message)
must be transformed into an iterable of text fragments with a small delay (`time.sleep(1/250)`)
to simulate the typing effect.

Since streamlit is by default stateless, all messages sent to the chatbot and its replies
are stored in the session state, and rewritten (not streamed) at the begining of each execution,
to keep the whole conversation in the screen.

All of this is performed by a custom function `add_message` that streams a message
the first time and stores it in the session state.

### Limitations

- Only the last message is actually sent to the LLM so even though the whole conversation is
displayed all the time, every query is independent. That is, the chatbot has no access to
the previous context. This is easy to fix by passing the last few messages on the call to `client.chat_stream(...)`.
- There is no caching of queries, only embeddings. The same query will consume API calls every time its used. This is relatively easy to fix by caching the query before submission, but you cannot simply use `st.cache_*` because the response is a stream of data objects that is consumed asyncronously, and the same query can have a different response per user/document.
- Only three chunks are retrieved for each query so if the question requires a longer context the model will give an incorrect answer.
- There is no attempt to sanitize the input or output, so the model can behave erratically and exhibit biased and impolite replies if prompted with such intention.

## Collaboration

Code is MIT. Fork, modify, and pull-request if you fancy :)
