from openai import OpenAI
from preprocess import *
import tiktoken
from dataset.get_dataset import get_instances
from seed_sampler import random_split, uniform_split
import json

import os
os.environ["OPENAI_API_KEY"]="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"

client = OpenAI()

def _get_seed_instances(seed_set_size, seed_set_generation_method=random_split, instance_function=get_instances):
    """Return labels corresponding to selected seed set and writes the seed set instances 
        to a text file to be fed into chatGPT-3.5"""
    x = instance_function()
    y = get_labels()
    if seed_set_generation_method == uniform_split:
        seed_x, seed_y = seed_set_generation_method(x, y, int(seed_set_size/7))
    else:
        seed_x, seed_y = seed_set_generation_method(x, y, seed_set_size)
        

    with open('COMP550-Research-Project/bootstrap_selected_instances.txt', 'w') as output_file:
        for line in seed_x:
            output_file.write(line + '\n')

        output_file.close()

    return seed_y


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

default_base = "For the following passage, tell me which domain the written text originates from out of the following possible domains:\
        Computer  Science, Electrical  Engineering,  Psychology,  Mechanical  Engineering, Civil  Engineering,  Medical  Science,  Biochemistry."

def _query_GPT_json(text_to_categorize, base_prompt=None, temperature=0, top_p=0.1, seed=1234):
    
    categories = ["Computer Science", "Electrical Engineering",  "Psychology",  
                  "Mechanical Engineering", "Civil Engineering",  "Medical Science",  "Biochemistry"]

    prompt = f"Respond in the json format: {{'response': text_categories}}\nText: {text_to_categorize}\nCategories (Computer Science, Electrical Engineering, Psychology, Mechanical Engineering, Civil Engineering, Medical Science, Biochemistry). Do not respond with categories outside of the ones stated."

    messages=[
        {"role": "system", "content": "You are a Natural Language Processing assistant, made to analyze text and categorize them based on their content."},
        {"role": "user", "content": prompt}
    ]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        messages=messages,
        max_tokens=40,
        n=1,
    )

    response_text =  completion.choices[0].message.content.strip()
    try:
        category = re.search("Computer Science|Electrical Engineering|Psychology|Mechanical Engineering|Civil Engineering|Medical Science|Biochemistry", response_text).group(0)
        category = categories.index(category) #convert to indices
    except AttributeError:
        category = '-1'
    # Add input_text back in for the result
    return str(category)

def _query_GPT(text_to_categorize, base_prompt=default_base, temperature=0, top_p=0.1, seed=1234):

    messages=[
        {"role": "system", "content": "You are a Natural Language Processing assistant, made to analyze text and categorize them based on their content."},
        {"role": "user", "content": f"{base_prompt} Passage to categorize: {text_to_categorize}"}
    ]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        messages=messages,
    )

    #print(f"Number of Tokens estimated for messages: {num_tokens_from_messages(messages, model='gpt-3.5-turbo')}")
    return completion.choices[0].message.content


def _generate_labels(base_prompt=default_base, query_method=_query_GPT, temperature=0, top_p=0.1, seed=1234, filepath_out='COMP550-Research-Project/gpt_generated_labels.txt', filepath_in='COMP550-Research-Project/bootstrap_selected_instances.txt'):
    """generate ChatGPT labels according to instances passed as input"""
    with open(filepath_in, 'r') as bootstrap_instances:
        gpt_labels=[]
        for instance in bootstrap_instances:
            label = query_method(instance, base_prompt=base_prompt, temperature=temperature, top_p=top_p, seed=seed) 
            gpt_labels.append(label)


        with open(filepath_out, 'a') as output_file:
            for label in gpt_labels:
                output_file.write(label + '\n')

            output_file.close()
        return gpt_labels


if __name__ == "__main__":
    print(_query_GPT_json(get_instances()[0]))
#    generate_labels()
# _get_seed_instances(seed_set_size=100)



