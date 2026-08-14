from dotenv import load_dotenv
load_dotenv()

# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

def main ():
    # OpenAI setup
    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_tokens=100)
    # response = llm.invoke("say 'setup complete' in one sentence")
    # print(response)
    # print(os.getenv("OPENAI_API_KEY"))

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    response = llm.invoke("Say setup complete in one sentence.")

    print(response.content)

if __name__ == "__main__":
    main()