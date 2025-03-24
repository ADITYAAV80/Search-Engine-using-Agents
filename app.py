import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
# from langchain_community.tools.asknews import AskNewsSearch
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
# from langchain.tools import BaseTool
# from typing import Any

load_dotenv()
# os.environ["ASKNEWS_CLIENT_ID"] = os.getenv("ASKNEWS_CLIENT_ID")
# os.environ["ASKNEWS_CLIENT_SECRET"] = os.getenv("ASKNEWS_CLIENT_SECRET")
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]



arxiv_api_wrapper = ArxivAPIWrapper(
    top_k_results=2,
    ARXIV_MAX_QUERY_LENGTH=200,
    load_max_docs=3,
    doc_content_chars_max=1500
)

wikipedia_api_wrapper = WikipediaAPIWrapper(
    top_k_results=2,
    lang = "en",
    doc_content_chars_max= 1500
)


arxiv_tool = ArxivQueryRun(api_wrapper = arxiv_api_wrapper)
wikipedia_tool = WikipediaQueryRun(api_wrapper = wikipedia_api_wrapper)
# name helps to distinguish two instances of same name
duckduck_tool  = DuckDuckGoSearchRun(name="duckduckgo_search")


# class WrappedNewsTool(BaseTool):
#     name: str = "wrapped_news"
#     description: str = "Fetches news from the last 24 hours."
#     def _run(self, *args: Any, **kwargs: Any) -> str:
#         news_tool = AskNewsSearch(max_results=4)
#         result = news_tool.run(query)
#         return str(result)

# wrapped_news_tool = WrappedNewsTool()

tools = [arxiv_tool,wikipedia_tool, duckduck_tool]
st.title("Welcome to Sastha Search Engine")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"ai","content":"Great Sir what thou search ?"}]

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

query = st.chat_input(placeholder="Search string here")


if query:

    st.chat_message("human").write(query)
    st.session_state["messages"].append({"role":"human","content":query})

    ###
    llm = ChatGroq(model="Llama3-8b-8192",streaming=True)
    search_agent= initialize_agent(tools=tools,llm=llm,agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors="Partial response: Here’s what I found before stopping...")
    ###

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
        response = search_agent.run({"input":query,"chat_history":st.session_state.messages}, callbacks=[st_cb])
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)
