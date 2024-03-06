import streamlit as st
import time
import torch
torch.cuda.empty_cache()
# torch.cuda.reset()
import chromadb
from types import SimpleNamespace
import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM



# Streamlit Page Title
st.write("# RAG Agents on Open Source LLM")

selected_model = None
selected_key = None

with st.sidebar:
    st.header("LLM Configuration")
    selected_model = st.selectbox("Model", ['Qwen/Qwen1.5-0.5B-Chat'], index=0)
    
    st.divider()
    
    st.write("# Project Links")
    st.write("All my ML project can be found on my GitHub profile: [link](https://github.com/IrshadG?tab=repositories)")
    st.write("My published android game on Google Play Store: [link](https://play.google.com/store/apps/details?id=com.LockdownProductions.PandemicRunner&pli=1)")

# For Autogen to recognize and load custom model from hf
class CustomModelClientWithArguments:
    def __init__(self, config, **kwargs):
        self.model_name = config["model"]
        self.model = loaded_model
        self.tokenizer = tokenizer

        self.device = config.get("device", "cpu")

        gen_config_params = config.get("params", {})
        self.max_length = gen_config_params.get("max_length", 256)
        print(f"Loaded model {config['model']} to {self.device}")

    def create(self, params):
        if params.get("stream", False) and "messages" in params:
            raise NotImplementedError("Local models do not support streaming.")
        else:
            num_of_responses = params.get("n", 1)
            response = SimpleNamespace()

            inputs = self.tokenizer.apply_chat_template(
                params["messages"], return_tensors="pt", add_generation_prompt=True
            ).to(self.device)
            inputs_length = inputs.shape[-1]

            # add inputs_length to max_length
            max_length = self.max_length + inputs_length
            generation_config = GenerationConfig(
                max_length=max_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            response.choices = []
            response.model = self.model_name

            for _ in range(num_of_responses):
                outputs = self.model.generate(inputs, generation_config=generation_config)
                # Decode only the newly generated text, excluding the prompt
                text = self.tokenizer.decode(outputs[0, inputs_length:])
                choice = SimpleNamespace()
                choice.message = SimpleNamespace()
                choice.message.content = text
                choice.message.function_call = None
                response.choices.append(choice)

            return response

    def message_retrieval(self, response):
        """Retrieve the messages from the response."""
        choices = response.choices
        return [choice.message.content for choice in choices]

    def cost(self, response) -> float:
        """Calculate the cost of the response."""
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response):
        # returns a dict of prompt_tokens, completion_tokens, total_tokens, cost, model
        # if usage needs to be tracked, else None
        return {}


with st.spinner():
    config = {
        "model": selected_model,
        "model_client_cls": "CustomModelClientWithArguments",
        "device": "cuda",
        "n": 2,
        "params": {
            "max_length": 1000,
        }
    },

    device = config[0].get("device", "cpu")
    loaded_model = AutoModelForCausalLM.from_pretrained(selected_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(selected_model, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm_config = config
    
    # create an AssistantAgent instance named "assistant"
    assistant = RetrieveAssistantAgent(name="assistant",
        system_message="You are Qwen Irshad's helpful AI assistant. You will answer any questions related to Irshad",
        llm_config={
            "timeout": 120,
            # "cache_seed": 42,
            "config_list": config,
        },)
    
    # Register model client
    assistant.register_model_client(
        model_client_cls=CustomModelClientWithArguments,
        loaded_model=loaded_model,
        tokenizer=tokenizer,
        )

    assistant.reset()
    # create a UserProxyAgent instance named "user"
    user_proxy = RetrieveUserProxyAgent(
        name="ragproxyagent",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=1,
        retrieve_config={
            "task": "code",
            "docs_path": [
                "./data/IrshadGirach.pdf",
            ],
            "chunk_token_size": 2000,
            "model": config[0]["model"],
            "client": chromadb.PersistentClient(path="/tmp/ig"),
            "embedding_model": "all-mpnet-base-v2",
            "get_or_create": True, 
        },
        code_execution_config=False,
    )
    
    # Accept user input
    entry_message = "Ask me a question"
    
    st.write("#### This is Qwen, my Resume assistant!")
    st.write("You can ask about me, my qualifications, projects or interests")
    st.warning("⚠️Please note that this project is running on local machine. Do not run it unless  you have access to a decent machine")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages = []    
    session_state_messages = []    
        
    placeholder = st.empty()
    input_state = False
    


_emp = st.empty()
my_human_input = _emp.chat_input("Enter something here!")
    
# The function that overrides the original function from autogen    
def get_human_input(prompt:str):
    global _emp
    global my_human_input   
    # my_human_input = _emp.chat_input("Enter Something here!", key=str(random.randrange(0,1000)))
    
    last_reply = user_proxy.last_message(assistant)['content']
    reply =  last_reply.replace('<|im_end|>', '')
    if 'UPDATE CONTEXT' in reply:
        reply = 'Ask me about Irshad.'
    session_state_messages.append({"role": 'assistant', "content": reply})
    
    with st.chat_message('assistant'):
        st.markdown(reply)
    # return input("what is input")   
    while True:
        if my_human_input or my_human_input!="":
            with st.chat_message('user'):
                    st.markdown(my_human_input)
            session_state_messages.append({"role": 'user', "content": my_human_input})
            my_human_input = ""
            # torch.cuda.empty_cache()
            # torch.cuda.reset()
            return session_state_messages[-1]['content']
        else:
            time.sleep(1)
 
# Override Function
user_proxy.get_human_input = get_human_input

# Start Chat
code_problem = "Hi"
user_proxy.initiate_chat(
    assistant, problem=code_problem
)


        
        
        