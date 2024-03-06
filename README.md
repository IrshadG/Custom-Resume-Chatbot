# Resume-Chatbot
 A custom chatbot that can answer anything about you from your CV!

## About
This project demonstrates the implementation of a personalized CV query system using an open-source model: Qwen 1.5-0.5B-Chat from [HuggingFace](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat), set up locally. This model was small enough to fit on my GPU. The [Autogen](https://github.com/microsoft/autogen) library by [Microsoft](https://opensource.microsoft.com/) is employed to orchestrate the LLM (Large Language Model) and apply Retrieve-Augmented-Generation (RAG). By uploading my resume details as a pdf to the LLM, the system can effectively answer any queries related to my CV, showcasing a potentially innovative approach to streamlining employee screenings.

In the near future, this technology could be harnessed by employers to upload and process large volumes of resumes simultaneously. The tool would enable the filtering and sorting of applicants based on their education, experience, skills, and projects, while also providing valuable insights through various statistics and visual representations. This innovation stands to significantly enhance the efficiency and effectiveness of the recruitment process.
<p float="left">
  <img src="images/LLM_ss_crop.png?raw=true" width="80%"/>
<p/>

 
## Process

#### 1. LLM Model
Due to my limited computing resources, I opted for the Qwen/Qwen1.5-0.5B-Chat model from HuggingFace, which has a relatively small parameter size of 0.5B yet performs adequately for my specific needs.

#### 2. Orchestration
Autogen offers an effective framework for managing and coordinating various LLMs. You can create multiple agents using these models and adjust them to do what you need through prompt engineering. In this project, I developed two RAG (Retrieve Augmented Generation) agents using Autogen: one that acts as a user-proxy, capable of reading documents and handling user's queries, and another agent that processes these queries and performs relevant operations. 
<p float="left">
  <img src="images/autogen_agentchat.png?raw=true" width="75%"/>
<p/>

#### 3. Interface & Output
Once all the components were in place, I utilized [Streamlit](https://streamlit.io/) to build a simple chat interface and connected it to my LLM orchestration. Although Streamlit provides a deployment option, I was unable to use it due to the hardware requirements needed to run the LLMs effectively.

![Chat](images/SS_LLM_2.png?raw=true)


## Acknowledgments

- Models infered/finetuned from [HuggingFace](https://huggingface.co/).

- [Autogen](https://github.com/microsoft/autogen) library used to orchestrate LLMs.

- Interface implemented using [Streamlit](https://streamlit.io/).

- Thanks to [Nazim Girach](https://github.com/ulfimlg) for contributing to this project.

