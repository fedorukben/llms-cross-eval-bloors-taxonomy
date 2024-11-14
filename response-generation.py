import time
from openai import OpenAI
import os
import requests
from enum import Enum
import google.generativeai as genai
import anthropic
from mistralai import Mistral
import cohere
import pickle

TEMPERATURE = 0.7

cohere_client = cohere.ClientV2(api_key=os.environ['COHERE_API_KEY'])
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
perplexity_key = os.environ['PERPLEXITY_API_KEY']
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
anthropic_client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
mistral_client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
grok_client = OpenAI(api_key=os.environ['XAI_API_KEY'], base_url="https://api.x.ai/v1")


class LLM(Enum):
    CHATGPT = 1
    CLAUDE = 2
    PERPLEXITY = 3
    GEMINI = 4
    MISTRAL = 5
    COHERE = 6
    GROK = 7

def ask(prompt, llm):
    if llm == LLM.CHATGPT:
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o-mini",
            temperature=TEMPERATURE
        )
        return response.choices[0].message.content
    elif llm == LLM.PERPLEXITY:
        url = "https://api.perplexity.ai/chat/completions"

        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": TEMPERATURE,
            "top_p": 0.9,
            "return_citations": True,
            "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "month",
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }
        headers = {
            "Authorization": f"Bearer {perplexity_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            
            # Extract the actual response content from the Perplexity API response
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0]['message']['content']
            else:
                raise Exception("No content in response")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Perplexity API error: {str(e)}")
    elif llm == LLM.GEMINI:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=TEMPERATURE))
        return response.text
    elif llm == LLM.CLAUDE:
        message = anthropic_client.messages.create(
            model = 'claude-3-5-sonnet-20241022',
            max_tokens=1000,
            temperature=TEMPERATURE,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ]
                }
            ]
        )
        return message.content[0].text
    elif llm == LLM.MISTRAL:
        response = mistral_client.chat.complete(
            model='mistral-large-latest',
            temperature=TEMPERATURE,
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )
        return response.choices[0].message.content
    elif llm == LLM.COHERE:
        response = cohere_client.chat(
            model="command-r-plus-08-2024",
            temperature=TEMPERATURE,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.message.content[0].text
    elif llm == LLM.GROK:
        response = grok_client.chat.completions.create(
            model="grok-beta",
            temperature=TEMPERATURE,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


experiments = {
    'remembering': [
        "List the main components of a plant cell and their primary functions.",
        "State Newton's three laws of motion.",
        "Define the term 'democracy' and identify its key principles.",
        "Name the major events that led to the French Revolution in chronological order.",
        "Recall the basic rules of subject-verb agreement in English grammar."
    ],
    'understanding': [
        "Explain how the water cycle works and its importance to Earth's ecosystem.",
        "Describe the relationship between supply, demand, and market prices.",
        "Summarize the main arguments in Martin Luther King Jr.'s 'I Have a Dream' speech.",
        "Compare and contrast mitosis and meiosis in cell division.",
        "Explain why the seasons change throughout the year using Earth's orbital patterns."
    ],
    'applying': [
        "Calculate the future value of $1000 invested at 5% compound itnerest over 10 years.",
        "Demonstrate how to balance this chemical equation: Na+Cl_2 -> NaCl.",
        "Apply the Pythagorean theorem to find the height of a ladder leaning against a wall.",
        "Show how Shakespeare's themes in 'Romeo and Juliet' relate to modern teenage relationships.",
        "Use the principles of operant conditioning to design a program to improve student homework completion."
    ],
    'analyzing': [
        "Analyze the impact of social media on political polarization in democratic societies.",
        "Differentiate between correlation and causation using the relationship between ice cream sales and swimming pool accidents.",
        "Examine how different factors contribute to climate change and their relative significance.",
        "Break down the effects of the Industrial Revolution on social class structure.",
        "Analyze the literary devices used in Edgar Allan Poe's 'The Raven' and their contribution to the poem's mood."
    ],
    'evaluating': [
        "Evaluate the effectiveness of remote leaerning versus traditional classroom instruction based on recent educational research.",
        "Assess the arguments for and against implementing a universal basic income.",
        "Judge the relative importance of nature versus nurture in personality development.",
        "Critique the current methods of plastic waste management and their long-term sustainability.",
        "Rate the success of different public health approaches to managing the COVID-19 pandemic."
    ],
    'creating': [
        "Design an experiment to test the effect of music on plant growth.",
        "Develop a solution to reduce food waste in urban areas using current technology.",
        "Create a marketing strategy for a new educational app targeting elementary school students.",
        "Propose a new system for public transportation in Toronto, Canada that addresses current urban mobility challenges.",
        "Write an alternative ending to 'The Great Gatsby' that aligns with the themes of the novel."
    ]
}

if __name__ == "__main__":
    total_iter_count = 0
    for level, questions in experiments.items():
        for question in questions:
            for llm in LLM:
                total_iter_count += 5

    active_iter_count = 0
    failed_iter_count = 0
    responses = []
    for level, questions in experiments.items():
        for question in questions:
            for _ in range(5):
                for llm in LLM:
                    try:
                        prompt = f"Capacity and Role: You are an expert educator responding to a {level} question. Approach this as if teaching a student while demonstrating the appropriate cognitive skills for this level. Insight: This question is designed to assess {level} capabilities according to the revised Bloom’s Taxonomy. Successful responses should be in paragraph form, without any formatting (bolding, italicizing, etc.). Statement: {question} Personality: Maintain a teaching style appropriate to the level. Speak with an appropriate level of formality. Write in paragraph form. Keep the response succinct yet complete. Experiment: Only output your response in your assigned capacity/role."
                        responses.append((ask(prompt, llm), llm, level, question))
                        active_iter_count += 1
                        print(f"Completed {active_iter_count}/{total_iter_count} tasks. {responses[-1][0][:20]}")
                    except Exception as e:
                        print(e)
                        print(f'FAILED with {llm} on level {level} and question {question}.')
                        failed_iter_count += 1
    print('Process Completed.')
    print(f'Successful: {active_iter_count}/{total_iter_count}.')
    print(f'Failed: {failed_iter_count}/{total_iter_count}.')
    
    output = open('response-generation.pkl', 'wb')
    pickle.dump(responses, output)
    output.close()
    print('Data pickled successfully.')