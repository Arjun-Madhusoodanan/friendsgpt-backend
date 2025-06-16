import pandas as pd
import re
from collections import Counter
from textblob import TextBlob
from openai import OpenAI
import random
import os
from dotenv import load_dotenv
import joblib  # <-- NEW

# Load API key
load_dotenv()
apiKey = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=apiKey)

# ----- Load Data ----- #
def load_data():
    FILE_PATH = "FriendsScript.csv"
    df = pd.read_csv(FILE_PATH)
    main_characters = [
        'Chandler Bing', 'Joey Tribbiani', 'Monica Geller',
        'Phoebe Buffay', 'Rachel Green', 'Ross Geller'
    ]
    character_dialogues = df[df['speaker'].isin(main_characters)]
    main_character_dialogues = character_dialogues.groupby('speaker')['text'].apply(
        lambda lines: [str(line) if isinstance(line, str) else '' for line in lines]
    )
    return main_character_dialogues, df

# ----- Analyze Character ----- #
def analyze_character(dialogues):
    all_text = ' '.join(dialogues)
    words = re.findall(r'\b\w+\b', all_text.lower())
    sentences = re.split(r'[.!?]+', all_text)

    word_count = len(words)
    unique_words = len(set(words))
    lexical_diversity = unique_words / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / len(sentences) if sentences else 0

    filtered_words = [w for w in words if len(w) > 3]
    common_words = Counter(filtered_words).most_common(10)

    sentiments = [TextBlob(line).sentiment.polarity for line in dialogues]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "lexical_diversity": round(lexical_diversity, 3),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_sentiment": round(avg_sentiment, 3),
        "top_words": common_words,
        "sentiments": sentiments
    }

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

# ----- Prompt Generator ----- #
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

# ----- Compute + Cache Stats ----- #
def compute_character_stats(dialogues_map):
    return {
        char: analyze_character(dialogues)
        for char, dialogues in dialogues_map.items()
    }

def compute_character_stats_cached():
    cache_file = "cached_character_stats.pkl"
    if os.path.exists(cache_file):
        return joblib.load(cache_file)
    else:
        dialogues, _ = load_data()
        stats = compute_character_stats(dialogues)
        joblib.dump(stats, cache_file)
        return stats

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

# ----- Entry Point ----- #
def run_friendsgpt(topic: str) -> str:
    dialogues, _ = load_data()
    character_stats = compute_character_stats_cached()
    prompt = generate_prompt(topic, character_stats, dialogues)
    return get_friends_response(prompt)
