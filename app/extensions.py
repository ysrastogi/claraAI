from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_core.prompts.prompt import PromptTemplate
# In-memory storage for simplicity
sessions = {}

# Initialize the LLM model
llm = Ollama(model="llama3")