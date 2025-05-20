-- USERS TABLE
CREATE TABLE fin_users (
    user_id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    surname TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    profile_pic TEXT 
);

-- CHAT HISTORY TABLE
CREATE TABLE chat_history (
    history_id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES fin_users(user_id),
    qa_pair JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
