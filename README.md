# Influencer Persona Profiler

A Streamlit-based app that analyzes a user's social media bio and recent posts to predict influencer personality types and recommend relevant product categories.

## Demo

https://weareerahadeera1.streamlit.app

---

## What It Does

- Analyzes bio and posts using Sentence Transformers
- Predicts top influencer personalityies using a combination of vector similarity and zero-shot classification
- Recommends products or sponsorship categories aligned with your persona
- Provides stylistic insights about your posting habits (e.g. emoji usage, post length)

---

## Example

**Bio:**
Reading minds and self-help books. Journaling my inner chaos. 🎭📚🧘

**Posts:**
Theatre rehearsal + breathwork = grounded chaos 🎤🫁
Just cried in public over a line in a novel 💔📖 #BookLover
Affirmation of the day: “I am not my email inbox.” 💻🙅‍♀️
Bubble bath + audiobook = ultimate reset 🛁🎧 #Wellness
Lit major turned empath. Send tea 🍵✨

---

## 🚀 Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/Hadeera3/weareera_task.git
   cd weareera_task
   ```

2. **Install Poetry**

   **macOS / Linux (zsh)**

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

   **Windows (PowerShell)**

   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   $env:Path += ";$env:USERPROFILE\.local\bin"
   ```

   Verify installation:

   ```bash
   poetry --version
   ```

3. **Install dependencies**

   ```bash
   poetry install --no-root
   ```

4. **Run the app**

   ```bash
   poetry run streamlit run app.py
   ```

## Project Structure

```
weareera_task/
├── app.py                       # Streamlit user interface
├── assign_scores.py             # Personality scoring & product recommendation logic
├── models.py                    # Model initialization & shared constants
├── insights.py                  # Post-processing analytics
├── data/
│   ├── personality_types_knowledge_base.json
│   └── product_catalog.json
├── pyproject.toml               # Poetry project configuration
├── poetry.lock                  # Locked dependency versions
└── README.md                    # Project documentation (you are here)
```

---

## Tech Stack

- Streamlit
- Sentence-Transformers
- scikit-learn
- Poetry

---

## To Do

- Scraping influencer pages instead of manually adding posts and bio
- Add metrics related to location
- Add metrics related to date
- Show progression of change in content
- Match brand products with influencers
