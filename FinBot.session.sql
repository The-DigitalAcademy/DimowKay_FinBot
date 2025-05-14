
-- USERS TABLE
CREATE TABLE users (
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
    user_id UUID REFERENCES users(user_id),
    qa_pair JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- SUGGESTED QUESTIONS TABLE
CREATE TABLE suggested_questions (
    id SERIAL PRIMARY KEY,
    question TEXT UNIQUE NOT NULL,
    answer TEXT NOT NULL
);





