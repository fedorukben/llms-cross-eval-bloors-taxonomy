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
import re

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
            "temperature": 0.2,
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
        response = model.generate_content(prompt)
        return response.text
    elif llm == LLM.CLAUDE:
        message = anthropic_client.messages.create(
            model = 'claude-3-5-sonnet-20241022',
            max_tokens=1000,
            temperature=0.2,
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
    pickle_input = open('responses.pkl', 'rb')
    responses = pickle.load(pickle_input)
    pickle_input.close()

    total_iter_count = len(responses)*6
    active_iter_count = 0
    failed_iter_count = 0
    evaluations = []
    for response, llm_curr, level, question in responses:
        for llm_peer in LLM:
            if not llm_peer == llm_curr: 
                prompt=f"""Capacity and Role: You are an experienced educational assessor. Your role is to evaluate the following responses from other LLMs acting as educators. 
                
                Insight: You are evaluating questions from different levels of the revised Bloom's taxonomy. You will not be given the specific level. You will use the following rubric: 
                
                Content Quality (25 points)
                Exceptional (21-25):

                Complete mastery of subject matter
                Highly accurate information
                Rich, relevant details and examples
                Comprehensive coverage of topic
                Seamless integration of concepts

                Proficient (16-20):

                Strong grasp of subject matter
                Generally accurate with minor errors
                Good use of details and examples
                Thorough coverage of main points
                Clear connection between concepts

                Developing (11-15):

                Basic understanding demonstrated
                Some accuracy with notable errors
                Limited details and examples
                Partial coverage of topic
                Basic connections between concepts

                Limited (6-10):

                Minimal understanding shown
                Significant errors present
                Few relevant details or examples
                Incomplete coverage
                Weak conceptual connections

                Insufficient (0-5):

                Little to no understanding
                Major inaccuracies throughout
                Lacks details and examples
                Very incomplete coverage
                No clear conceptual connections

                Cognitive Alignment (25 points)
                Exceptional (21-25):

                Perfect alignment with intended cognitive level
                Demonstrates sophisticated thinking at target level
                Exceeds cognitive requirements
                Shows mastery of required mental processes
                Exemplary demonstration of level-appropriate skills

                Proficient (16-20):

                Good alignment with cognitive level
                Shows appropriate thinking for level
                Meets cognitive requirements
                Demonstrates required mental processes
                Clear evidence of level-appropriate skills

                Developing (11-15):

                Partial alignment with cognitive level
                Inconsistent thinking at target level
                Partially meets requirements
                Some evidence of required processes
                Basic level-appropriate skills shown

                Limited (6-10):

                Minimal alignment with cognitive level
                Often below target thinking level
                Rarely meets requirements
                Limited evidence of required processes
                Few level-appropriate skills demonstrated

                Insufficient (0-5):

                No alignment with cognitive level
                Below target thinking level
                Does not meet requirements
                No evidence of required processes
                Lacks level-appropriate skills

                Communication Effectiveness (25 points)
                Exceptional (21-25):

                Crystal clear explanation
                Excellent organization
                Precise terminology use
                Highly engaging presentation
                Perfect formatting and structure

                Proficient (16-20):

                Clear explanation
                Good organization
                Appropriate terminology
                Engaging presentation
                Proper formatting and structure

                Developing (11-15):

                Somewhat clear explanation
                Basic organization
                Some appropriate terminology
                Moderately engaging
                Basic formatting and structure

                Limited (6-10):

                Unclear explanation
                Poor organization
                Inappropriate terminology
                Not engaging
                Inconsistent formatting

                Insufficient (0-5):

                Very unclear explanation
                No organization
                Missing or wrong terminology
                Confusing presentation
                Poor formatting

                Response Development (25 points)
                Exceptional (21-25):

                Comprehensive development
                Excellent support/evidence
                Sophisticated reasoning
                Creative/innovative elements
                Outstanding depth of analysis

                Proficient (16-20):

                Good development
                Solid support/evidence
                Clear reasoning
                Some creative elements
                Good depth of analysis

                Developing (11-15):

                Basic development
                Some support/evidence
                Basic reasoning
                Few creative elements
                Limited analysis

                Limited (6-10):

                Minimal development
                Little support/evidence
                Weak reasoning
                No creative elements
                Superficial analysis

                Insufficient (0-5):

                No development
                No support/evidence
                No clear reasoning
                Lacks creativity
                No analysis

                Statement: Your task is to evaluate the following response: {response}

                Personality: You should be brief and concise, keeping a professional tone.  

                Experiment: Your format consists of two parts. First, you should give an integer between 0 and 100 (inclusive) representing the total score out of 100, with no punctuation. Then, put a * as a separator. Then write 1 sentence of brief and concise feedback. The first part should be purely an integer, as it is going to be parsed by code. It must be purely an integer value with no text. Do not use any special typefaces (e.g., bold, italics, etc.).
                """
                try:
                    evaluation = ask(prompt, llm_peer)
                    match = re.search(r'\d+', evaluation)
                    if match:
                        score = int(match.group())
                    else:
                        print("ERROR: NO SCORE GIVEN!!!")
                        score = ""
                    evaluations.append((score, evaluation, llm_peer, llm_curr, response, level, question))
                    active_iter_count += 1
                    print(f"Completed {active_iter_count}/{total_iter_count} tasks. Score: {score}")
                except Exception as e:
                    print(e)
                    print(f'FAILED with {llm_peer} from {llm_curr} and its response {response}')
                    failed_iter_count += 1
    print('Process Completed.')
    print(f'Successful: {active_iter_count}/{total_iter_count}.')
    print(f'Failed: {failed_iter_count}/{total_iter_count}.')

    pickle_output = open('crossevals.pkl', 'wb')
    pickle.dump(evaluations, pickle_output)
    pickle_output.close()
    print('Data pickled successfully.')