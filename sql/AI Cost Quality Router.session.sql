-- Create Data Base and Setup SQL Tools
CREATE DATABASE ai_router;
SELECT current_database(), current_user;

SELECT datname
FROM pg_database
WHERE datname = 'ai_router';