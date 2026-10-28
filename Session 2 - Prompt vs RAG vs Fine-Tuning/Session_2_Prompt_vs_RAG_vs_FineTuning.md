# Session 2: Prompt Engineering vs. RAG vs. Fine-Tuning

This session explores the architectural trade-offs among the three primary paradigms for customizing Large Language Model behavior: In-Context **Prompt Engineering**, Dynamic **Retrieval-Augmented Generation (RAG)**, and Parametric **Fine-Tuning**.

---

## Task 1: Instagram Bio Generation via Prompt Engineering

### Problem Statement
Write a prompt for ChatGPT that helps quickly generate a catchy Instagram bio for a college student, test it, and record the result.

---

### Engineered Prompt
```text
Generate 3 catchy, aesthetic Instagram bios for a 20-year-old computer engineering college student living in Ahmedabad.
Interests: Python coding, street photography in the Old City, and late-night filter coffee.

Constraints:
1. Each bio must be under 150 characters (Instagram bio limit).
2. Use clean formatting with line breaks and relevant emojis.
3. Include a clever, relatable student hook or call-to-action.
```

#### Actual Generated Output
> **Option 1 (Tech & Coffee Lover):**  
> `Debugging code by day, chasing sunsets by night 🌇`  
> `Ahmedabad streets through my 35mm lens 📸`  
> `Fuelled by filter coffee & syntax errors ☕💻`  
> `👇 Check out my latest photo series`  
> 
> **Option 2 (Minimalist Aesthetic):**  
> `turning coffee into code & chaos into frames ☕✨`  
> `comp sci '26 | capturing the pols of Amdavad 🏛️`  
> `Ctrl + S your memories 📸`  
> 
> **Option 3 (Witty Student Persona):**  
> `Binary thoughts in a heritage city 💻🕌`  
> `Engineering student by compulsion, photographer by passion 🎞️`  
> `Probably at a tapri sipping chai ☕👇`  

---

## Task 2: Simulated RAG Workflow (IPL Cricket Statistics)

### Problem Statement
Simulate a Retrieval-Augmented Generation (RAG) workflow in Python by reading an external factual text file containing IPL statistics, performing keyword/semantic retrieval matching the user query, and generating an augmented response combining the user's question with retrieved context.

---

### Implementation Architecture
- **Knowledge Base File:** [`ipl_stats.txt`](file:///C:/Users/DHRUV%20MODI/Documents/TOPS/assignment/FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION/%20FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION%20Assignment/Session%202%20-%20Prompt%20vs%20RAG%20vs%20Fine-Tuning/ipl_stats.txt)
- **Python Script:** [`ipl_rag_simulation.py`](file:///C:/Users/DHRUV%20MODI/Documents/TOPS/assignment/FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION/%20FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION%20Assignment/Session%202%20-%20Prompt%20vs%20RAG%20vs%20Fine-Tuning/ipl_rag_simulation.py)

#### Console Execution Output:
```text
IPL RETRIEVAL-AUGMENTED GENERATION (RAG) SIMULATION ENGINE

===========================================================================
USER QUESTION: Who holds the record for the most career runs in IPL history?
===========================================================================
Query: Who holds the record for the most career runs in IPL history?
Retrieved Factual Knowledge:
- Record 1: Most Career Runs in IPL - Virat Kohli leads the all-time scoring charts with 8004 runs in 252 matches at an average of 38.66 and a strike rate of 131.97, including 8 centuries and 55 half-centuries.

RAG Augmented Synthesis:
Based on the retrieved tournament records, Record 1: Most Career Runs in IPL - Virat Kohli leads the all-time scoring charts with 8004 runs in 252 matches at an average of 38.66 and a strike rate of 131.97, including 8 centuries and 55 half-centuries.

===========================================================================
USER QUESTION: What is the highest team total ever scored in the IPL?
===========================================================================
Query: What is the highest team total ever scored in the IPL?
Retrieved Factual Knowledge:
- Record 3: Highest Team Total - Sunrisers Hyderabad scored the highest team total in IPL history, posting 287/3 in 20 overs against Royal Challengers Bengaluru at M. Chinnaswamy Stadium on April 15, 2024.

RAG Augmented Synthesis:
Based on the retrieved tournament records, Record 3: Highest Team Total - Sunrisers Hyderabad scored the highest team total in IPL history, posting 287/3 in 20 overs against Royal Challengers Bengaluru at M. Chinnaswamy Stadium on April 15, 2024.

===========================================================================
USER QUESTION: Who won the IPL 2024 championship title?
===========================================================================
Query: Who won the IPL 2024 championship title?
Retrieved Factual Knowledge:
- Record 10: IPL 2024 Champions - Kolkata Knight Riders won the IPL 2024 championship, defeating Sunrisers Hyderabad by 8 wickets in the final at MA Chidambaram Stadium, Chennai.

RAG Augmented Synthesis:
Based on the retrieved tournament records, Record 10: IPL 2024 Champions - Kolkata Knight Riders won the IPL 2024 championship, defeating Sunrisers Hyderabad by 8 wickets in the final at MA Chidambaram Stadium, Chennai.
```

---

## Task 3: Real-World Scenarios for Prompting, RAG, and Fine-Tuning

### 1. Prompt Engineering: Zomato Festive Tone & Greeting Customization
* **Scenario:** Adapting Zomato's conversational assistant to adopt festive personas (e.g., cheerful Diwali wishes, witty cricket banter during IPL matches, or Navratri fasting recommendations).
* **Why Prompt Engineering is Best:** The task requires no proprietary internal data access or modified model weights; it only requires conditioning the conversational style and greeting format. Changes take minutes to test and deploy without infrastructure overhead.

### 2. RAG: Flipkart Real-Time Catalog & Price Discovery Assistant
* **Scenario:** A shopping assistant answering queries like *"Which 5G phones under ₹25,000 have an AMOLED screen, are in stock in Ahmedabad pin code 380015, and have active HDFC credit card discounts?"*
* **Why RAG is Best:** E-commerce inventory, flash discounts, delivery timelines, and warehouse stock change thousands of times per minute. Fine-tuning cannot accommodate real-time fluctuating database records. RAG retrieves live metadata from Elasticsearch/vector catalogs and grounds the LLM response in 100% verified stock data.

### 3. Fine-Tuning: Paytm Fraud Transaction & Phishing Intent Classifier
* **Scenario:** An automated real-time transaction screening engine that evaluates payment remarks, merchant descriptions, and dispute claims to detect emerging UPI scam patterns, account takeovers, and fraudulent payment solicitations.
* **Why Fine-Tuning is Best:** The system requires sub-50ms latency, deterministic classification labels (Fraud vs. Genuine), extreme privacy (financial transaction data cannot be leaked through prompt injection), and zero tolerance for prompt hallucination. Embedding fraud pattern detection directly into model weights via fine-tuning provides superior accuracy, speed, and security.

---

## Task 4: Comparison Table: Prompting vs. RAG vs. Fine-Tuning

| Evaluation Criteria | Prompt Engineering | Retrieval-Augmented Generation (RAG) | Model Fine-Tuning (PEFT / Full) |
| :--- | :--- | :--- | :--- |
| **Setup Cost** | **Negligible ($0 - Low):** Requires only standard API access fees and developer time for prompt iteration. | **Moderate:** Requires embedding models, vector database hosting (e.g., Pinecone/Chroma), chunking pipelines, and search indexing. | **High to Moderate:** Requires curated labeled datasets, GPU compute clusters (A100/H100), and ML engineering talent. |
| **Time to Deploy** | **Minutes to Hours:** Can be crafted, tested, and shipped immediately via API or system prompt updates. | **Days to Weeks:** Involves building document ingestion pipelines, indexing chunk hierarchies, and tuning retrieval recall. | **Weeks to Months:** Involves data curation, cleaning, hyperparameter tuning, validation runs, and regression testing. |
| **Maintenance Effort** | **Low:** Update prompt templates as instructions or business requirements evolve. | **Moderate:** Continuous document synchronization, index updating, re-ranking optimization, and handling out-of-date vector chunks. | **High:** Retraining or updating LoRA adapters when base model weights or core task definitions change. |
| **Data Freshness** | Dependent on model knowledge cutoff (static) unless manually passed in context. | **Dynamic & Real-Time:** Instantly reflects any newly added document or live database record without retraining. | **Static:** Model weights represent a snapshot in time; requires new fine-tuning runs to learn fresh information. |
| **Domain Adaptation** | Surface-level; guides tone, role, formatting, and high-level behavioral constraints. | High factual grounding through external context injection; does not change fundamental language style. | **Deep Behavioral & Syntax Adaptation:** Permanently modifies language style, classification accuracy, and domain terminology. |
| **Ideal Use Cases** | Brainstorming, ad copy generation, summary translation, structured formatting. | Enterprise knowledge search, customer support manuals, real-time product catalogs, legal search. | Specialized domain tasks (medical diagnoses, code generation, sentiment scoring, strict compliance). |

---

## Task 5: Decision Advice for a BookMyShow-Style Movie Recommendation Chatbot

### Consultation Prompt Given to ChatGPT:
```text
I am designing an intelligent movie recommendation and ticketing chatbot for a BookMyShow-style application. 
The chatbot needs to recommend movies based on genre and user mood, answer questions about cinema showtimes and seat availability in specific cities, and provide reviews/ratings.

Which approach should I use: Prompt Engineering, RAG, or Fine-Tuning? Provide your architectural recommendation with reasons.
```

#### Advice Received from ChatGPT:
> *"For a BookMyShow-style movie recommendation and ticketing chatbot, **a Hybrid Architecture combining RAG with Prompt Engineering is the strongly recommended solution**:
>
> 1. **Why NOT Fine-Tuning alone:** Cinema showtimes, theatre schedules, running movie lists, and available seat matrices change hourly. A fine-tuned model cannot store live showtimes in its weights and will hallucinate non-existent movie screenings.
> 2. **Why RAG is Essential:** Movie databases (synopses, cast details, IMDb/Rotten Tomatoes ratings) and live booking APIs (theatres, showtimes, ticket prices, seat layouts) must be dynamically retrieved and injected into the conversational prompt.
> 3. **Why Prompt Engineering completes the system:** A well-crafted system prompt provides the polite, energetic cinema concierge persona, instructs the model how to format showtime options into clean lists, and enforces ticketing conversion hooks.
>
> **Final Recommendation:** Use Prompt Engineering for conversational tone and user preference elicitation, combined with RAG connected to your live movie catalog and cinema ticketing API."*

---

### My Final Engineering Decision:
**I concur entirely with the Hybrid RAG + Prompt Engineering decision.**  

Movie ticketing platforms operate in an environment characterized by extreme temporal volatility—new showtimes drop daily, seats sell out in minutes, and movie releases change every Friday. Relying on model fine-tuning for movie availability is technically flawed because parametric weights cannot track dynamic state. 

Therefore, our production architecture will implement:
- **System Prompt:** Sets the persona of an enthusiastic cinema curator and enforces strict JSON output for booking hand-offs.
- **RAG Layer:** Connects to a Vector Database for semantic genre matching (*"Give me a psychological thriller like Inception"*) and integrates live REST APIs for theatre seat availability and showtime schedules.
