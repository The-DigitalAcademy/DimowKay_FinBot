import psycopg2

# --- PostgreSQL DB connection ---
def get_connection():
    return psycopg2.connect(
        host="localhost",         
        database="finbot",     
        user="postgres",          
        password="none"  
    )

# --- Financial questions and answers ---
faq_data = [
    ("What is a credit score?",
     "A credit score is a number ranging from 300 to 850 that reflects your creditworthiness. "
     "Lenders use it to evaluate how likely you are to repay debt. A higher score (above 700) "
     "typically indicates good credit and can lead to better loan terms."),

    ("How do I save for retirement?",
     "Start early, even with small amounts. Use retirement accounts like a pension fund or retirement annuity. "
     "Invest consistently, and take advantage of employer-matching contributions if available. Time and compound interest help grow your savings."),

    ("What is compound interest?",
     "Compound interest is when you earn interest not only on your initial deposit but also on the interest that has been added over time. "
     "Example: R1,000 earning 5% annually grows to R1,050 in one year, and R1,102.50 the next year."),

    ("How can I reduce my debt?",
     "Make a list of all debts, pay more than the minimum on high-interest ones, and avoid taking on new debt. "
     "Try the snowball method (smallest debt first) or avalanche method (highest interest first)."),

    ("How do I create a budget?",
     "Track your income and expenses. Categorize spending (needs, wants, savings). Set limits and stick to them. "
     "Apps like 22seven or Excel sheets help. Adjust monthly and save consistently."),

    ("What is an emergency fund?",
     "An emergency fund is savings set aside to cover unexpected expenses (e.g., medical bills, car repairs). "
     "Aim for 3–6 months of living expenses in a separate, easily accessible account."),

    ("How do interest rates work?",
     "Interest rates determine how much you'll earn on savings or pay on loans. "
     "Higher rates mean more earnings on investments but also more cost on debt. "
     "They're usually set by the central bank and influenced by inflation."),

    ("What's the difference between a debit and credit card?",
     "A debit card uses your own money from your bank account. A credit card borrows money from the bank that you must repay later, often with interest. "
     "Debit avoids debt; credit can build your credit score if managed responsibly."),

    ("How do I start investing with little money?",
     "Use platforms like EasyEquities or TymeInvest. Start with low-cost ETFs or unit trusts. "
     "Invest consistently, even with as little as R100/month. Focus on long-term growth rather than quick returns."),

    ("What are the best budgeting strategies?",
     "Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings/debt. Automate savings, use budgeting apps, and review spending monthly. "
     "Cut non-essential expenses and set financial goals.")
]

# --- Insert FAQ into the database ---
def insert_faq_data():
    conn = get_connection()
    cur = conn.cursor()
    for question, answer in faq_data:
        try:
            cur.execute("""
                INSERT INTO suggested_questions (question, answer)
                VALUES (%s, %s)
                ON CONFLICT (question) DO NOTHING;
            """, (question, answer))
        except Exception as e:
            print(f"Error inserting '{question}': {e}")
    conn.commit()
    cur.close()
    conn.close()
    print("FAQs inserted successfully.")

# --- Load only questions from DB ---
def load_suggested_questions():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT question FROM suggested_questions")
    questions = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return questions

# --- Get answer for a specific question ---
def get_answer(question, user_id=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT answer FROM suggested_questions WHERE question = %s", (question,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result:
        return result[0]
    return "Sorry, I don't have an answer for that yet."

# --- Run this file directly to insert data ---
if __name__ == "__main__":
    insert_faq_data()
