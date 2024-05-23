from langchain_community.llms import Bedrock

def get_text_response(input_content): #text-to-text client function

    llm = Bedrock( #create a Bedrock llm client
        model_id="mistral.mixtral-8x7b-instruct-v0:1", #set the foundation model
        model_kwargs={ #Mistral
            "max_tokens" : 512,
            "stop" : [],    
            "temperature": 0,
            "top_p": 0.9,
            "top_k": 50
        }
    )
    
    return llm.invoke(f"[INST]{input_content}[/INST]") #return a response to the prompt


