"""
Shared utility functions and sample texts for the summarisation app.

Nothing ML-specific lives here — just plain-Python helpers and the demo
texts shown in the sidebar dropdown.
"""

from __future__ import annotations

import io
from PyPDF2 import PdfReader


# ── File reading ───────────────────────────────────────────────────

def extract_text_from_file(file_obj, filename: str) -> str:
    """Extract and return text from an uploaded Streamlit file object."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(file_obj)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    else:
        # assume text file
        return file_obj.getvalue().decode("utf-8", errors="ignore")


# ── Text metrics ───────────────────────────────────────────────────

def word_count(text: str) -> int:
    """Return the number of whitespace-delimited words in *text*."""
    if not text:
        return 0
    return len(text.split())


def char_count(text: str) -> int:
    """Return the total number of characters (including spaces)."""
    return len(text)


def sentence_count(text: str) -> int:
    """
    Quick sentence count using NLTK's Punkt tokenizer.

    Importing here avoids a circular dependency — the rest of
    ``preprocess`` isn't needed.
    """
    if not text or not text.strip():
        return 0
    from src.preprocess import split_sentences
    return len(split_sentences(text))


def reading_time_seconds(text: str, wpm: int = 200) -> int:
    """
    Estimate reading time in seconds, assuming *wpm* words-per-minute.

    The average adult reads ~200-250 wpm; we default to the lower end
    to be conservative.
    """
    words = word_count(text)
    if words == 0:
        return 0
    return max(1, round(words / wpm * 60))


# ── Sample texts shown in the Streamlit sidebar ───────────────────

SAMPLE_TEXTS: dict[str, str] = {
    "Artificial Intelligence": (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "as opposed to natural intelligence displayed by animals including humans. "
        "AI research has been defined as the field of study of intelligent agents, "
        "which refers to any system that perceives its environment and takes actions "
        "that maximize its chance of achieving its goals. The term artificial intelligence "
        "had previously been used to describe machines that mimic and display human "
        "cognitive skills that are associated with the human mind, such as learning and "
        "problem-solving. Major AI researchers now reject this definition of AI and describe "
        "AI in terms of rationality and acting rationally, which does not limit how "
        "intelligence can be articulated. AI applications include advanced web search engines, "
        "recommendation systems, understanding human speech, self-driving cars, automated "
        "decision-making, and competing at the highest level in strategic game systems. "
        "As machines become increasingly capable, tasks considered to require intelligence "
        "are often removed from the definition of AI. For instance, optical character "
        "recognition is frequently excluded from things considered to be AI, having become "
        "a routine technology. Artificial intelligence was founded as an academic discipline "
        "in 1956, and in the years since it has experienced several waves of optimism, "
        "followed by disappointment and the loss of funding, followed by new approaches, "
        "success, and renewed funding."
    ),
    "Climate Change": (
        "Climate change refers to long-term shifts in temperatures and weather patterns. "
        "These shifts may be natural, but since the 1800s, human activities have been the "
        "main driver of climate change, primarily due to burning fossil fuels like coal, oil, "
        "and gas. Burning fossil fuels generates greenhouse gas emissions that act like a "
        "blanket wrapped around the Earth, trapping the sun's heat and raising temperatures. "
        "Examples of greenhouse gas emissions that are causing climate change include carbon "
        "dioxide and methane. These come from using gasoline for driving a car or coal for "
        "heating a building. Clearing land and forests can also release carbon dioxide. "
        "Landfills for garbage are a major source of methane emissions. Energy, industry, "
        "transport, buildings, agriculture, and land use are among the main emitters. "
        "The consequences of climate change now include intense droughts, water scarcity, "
        "severe fires, rising sea levels, flooding, melting polar ice, catastrophic storms, "
        "and declining biodiversity. People are experiencing climate change in diverse ways. "
        "Climate change can affect our health, ability to grow food, housing, safety, and work. "
        "Some of us are already more vulnerable to climate impacts, such as people living in "
        "small island nations. Conditions like sea-level rise and saltwater intrusion have "
        "advanced to the point where whole communities have had to relocate. In the future, "
        "the number of climate refugees is expected to rise."
    ),
    "Machine Learning": (
        "Machine learning is a subset of artificial intelligence that provides systems the "
        "ability to automatically learn and improve from experience without being explicitly "
        "programmed. Machine learning focuses on the development of computer programs that "
        "can access data and use it to learn for themselves. The process of learning begins "
        "with observations or data, such as examples, direct experience, or instruction. "
        "It looks for patterns in data and makes better decisions in the future based on the "
        "examples that we provide. The primary aim is to allow the computers to learn "
        "automatically without human intervention or assistance and adjust actions accordingly. "
        "Machine learning algorithms are often categorized as supervised, unsupervised, or "
        "reinforcement learning. Supervised learning algorithms apply what has been learned "
        "in the past to new data using labeled examples to predict future events. Unsupervised "
        "learning algorithms are used when the information used to train is neither classified "
        "nor labeled. Reinforcement learning interacts with its environment by producing "
        "actions and discovers errors or rewards. Deep learning is one of the most popular "
        "types of machine learning algorithms. It uses neural networks with many layers to "
        "learn complex patterns in large amounts of data. Common applications include image "
        "recognition, speech recognition, and natural language processing."
    ),
}