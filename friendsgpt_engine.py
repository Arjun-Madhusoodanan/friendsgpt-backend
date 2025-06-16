# friendsgpt_engine.py

import re
import json
import random
from openai import OpenAI
from dotenv import load_dotenv
import os
from textblob import TextBlob

# Load environment variables
load_dotenv()
apiKey = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=apiKey)

# ----- Load Precomputed Character Stats and Dialogues ----- #
def load_cached_data():
    with open("precomputed_friends_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    character_stats = {}
    dialogues = {}

    for character, details in data.items():
        character_stats[character] = {
            k: v for k, v in details.items() if k != "dialogues"
        }
        dialogues[character] = details["dialogues"]

    return character_stats, dialogues

# ----- Style Summary for Topic ----- #
def style_summary_topic(dialogues, topic):
    topic_lines = [line for line in dialogues if topic.lower() in line.lower()]
    if not topic_lines:
        topic_lines = dialogues

    exclam = sum(1 for line in topic_lines if "!" in line)
    quest = sum(1 for line in topic_lines if "?" in line)
    short = sum(1 for line in topic_lines if len(line.split()) < 4)

    form = "short, punchy" if short > len(topic_lines) * 0.4 else "long-winded"

    sentiments_filtered = [TextBlob(line).sentiment.polarity for line in topic_lines]
    avg_sent = sum(sentiments_filtered) / len(sentiments_filtered) if sentiments_filtered else 0

    if avg_sent >= 0.2:
        tone = "cheerful"
    elif avg_sent >= 0.05:
        tone = "light-hearted"
    elif avg_sent <= -0.2:
        tone = "pessimistic"
    elif avg_sent <= -0.05:
        tone = "cynical"
    else:
        tone = "neutral"

    question_tendency = "asks lots of questions" if quest > exclam else "makes bold statements"

    return f"Speaks in {form} lines about this topic, is generally {tone}, and {question_tendency}."

# ----- Prompt Generation ----- #
def generate_prompt(user_input, character_stats, full_dialogues):
    prompt = "Simulate a group chat between the 6 Friends characters."
    prompt += f"\nThe topic is: '{user_input}'\n"
    prompt += "Use each character's actual tone and style when talking about this topic.\n"

    topic_keywords = [word.lower() for word in re.findall(r'\w+', user_input)]

    for character, stats in character_stats.items():
        sample_words = ', '.join([word for word, _ in stats["top_words"][:5]])
        style = style_summary_topic(full_dialogues[character], user_input)

        topic_relevant_lines = [
            line for line in full_dialogues[character]
            if any(keyword in line.lower() for keyword in topic_keywords)
        ]
        if not topic_relevant_lines:
            continue
        examples = random.sample(topic_relevant_lines, min(2, len(topic_relevant_lines)))

        prompt += f"\n{character}: Uses words like [{sample_words}]. {style} Example lines: {' | '.join(examples)}"

    prompt += "\nNow generate the conversation with one message from each character who had relevant interactions."
    return prompt

# ----- OpenAI Chat Call ----- #
def get_friends_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're simulating a group chat among the Friends characters."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8
    )
    return response.choices[0].message.content

# ----- Main Execution Function ----- #
def run_friendsgpt(topic: str) -> str:
    character_stats, dialogues = load_cached_data()
    prompt = generate_prompt(topic, character_stats, dialogues)
    return get_friends_response(prompt)
