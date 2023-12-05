from openai import OpenAI
from preprocess import *
import tiktoken

import os
os.environ["OPENAI_API_KEY"]="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"

client = OpenAI()

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

def query_GPT(prompt):

    messages=[
        {"role": "system", "content": "You are a Natural Language Processing assistant, made to analyze text and categorize them based on their content."},
        {"role": "user", "content": f"For the following passage, tell me which domain the written text originates from out of the following possible domains:\
        Computer  Science, Electrical  Engineering,  Psychology,  Mechanical  Engineering, Civil  Engineering,  Medical  Science,  Biochemistry. Format your reply \
        as a single number, ranging from 0 to 6, with the number corresponding to the index of the domain mentioned above. Return no text, only the single number. Passage to categorize: {prompt}"}
    ]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    print(f"Number of Tokens estimated for messages: {num_tokens_from_messages(messages, model='gpt-3.5-turbo')}")
    return completion.choices[0].message.content





sample_feature = "Background: Chronic alcohol intake impacts skin directly, through organ dysfunction or by modifying preexisting dermatoses. \
                However, dermatoses afflicting chronic alcoholics figure in a few studies only. Aim: This study aims to correlate the spectrum \
                of dermatoses in chronic alcoholics with the quantum/duration of alcohol intake and raised liver transaminases. Materials and \
                Methods: Adult males, totaling 196, ascertained to fulfill the Royal College of Psychiatry criteria for chronic alcoholism by \
                the de -addiction center and referred for dermatological consult were enrolled as cases, and similar number of age -/sex -matched \
                teetotallers, as controls. Data emanating from detailed history, clinical examination, and routine liver functions tests were summarized \
                and subsequently analyzed, including statistically using the Chi-square, independent t and Spearman's rank correlation tests, and compared\
                with data from previous studies. Results: Majority (104) drank 41-50 units of alcohol/week since 3-40 (mean: 20.01 +/- 9.322) years. \
                Generalized pruritus (odds ratio [OR]: 31.15, P < 0.001), xerosis (OR: 3.62, P = 0.008), and seborrheic dermatitis (OR: 12.26, P < 0.001)\
                were significantly more common in cases than controls. Infections (73; 37.2%), eczemas (45; 22.9%), and generalized hyperpigmentation (28; 14.2%)\
                were- the major presenting complaints. Spider nevi, gynecomastia, and pellagroid dermatitis were present in 34 (17.3%), 19 (9.7%), and 8 (4.1%) \
                respectively exclusively in cases only. Commonly seen systemic abnormalities were an alcoholic liver disease (45; 22.9%), diabetes mellitus (23; 11.7%)\
                , and peripheral neuropathy (19; 9.7%). Conclusion: Knowledge of cutaneous manifestations of chronic alcoholism could prompt in-depth history taking of \
                alcohol intake, lead to specialist referral and thereby enable timely de -addiction, hopefully before serious adversities in the chronic alcoholics."
print(query_GPT(sample_feature))