# generate_character_cache.py

import pandas as pd
import pickle
from friendsgpt_engine import load_data, compute_character_stats
import re
import json
from collections import Counter
from textblob import TextBlob

if __name__ == "__main__":
    print("Loading data...")
    main_character_dialogues, _ = load_data()

    print("Computing character stats...")
    stats = compute_character_stats(main_character_dialogues)

    with open("cached_character_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    print("✅ Cache saved as cached_character_stats.pkl")

import pandas as pd
import re
import json
from collections import Counter
from textblob import TextBlob

# Load CSV
df = pd.read_csv("FriendsScript.csv")

main_characters = [
    'Chandler Bing', 'Joey Tribbiani', 'Monica Geller',
    'Phoebe Buffay', 'Rachel Green', 'Ross Geller'
]

character_dialogues = df[df['speaker'].isin(main_characters)]
main_character_dialogues = character_dialogues.groupby('speaker')['text'].apply(
    lambda lines: [str(line) if isinstance(line, str) else '' for line in lines]
)

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
        "sentiments": sentiments,
        "dialogues": dialogues  # needed for topic match later
    }

character_stats = {
    character: analyze_character(dialogues)
    for character, dialogues in main_character_dialogues.items()
}

# Save to JSON
with open("precomputed_friends_data.json", "w", encoding='utf-8') as f:
    json.dump(character_stats, f, indent=2, ensure_ascii=False)

print("✅ Precomputed data saved to precomputed_friends_data.json")