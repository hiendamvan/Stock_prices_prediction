import google.generativeai as genai 
import os 
from dotenv import load_dotenv 
import pandas as pd 
import time 
import random 

load_dotenv() 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) 
llm = genai.GenerativeModel("gemini-2.0-flash") 

def sentiment(text, timeout = 5, attempts = 3):    
    prompt = f"""
You are a financial sentiment analysis model.

Your task: Analyze the sentiment of the following Vietnamese financial news content and return **only one number** representing the sentiment score in the range (-1 to 1).

Rules:
- 1 = very positive (bullish)
- 0 = neutral
- -1 = very negative (bearish)
- Understand context of stock market terminology (tăng, giảm, điều chỉnh, hồi phục, áp lực bán, khối ngoại mua ròng,...)
- Ignore numbers, dates, and objective statements unless they imply positive/negative market sentiment.
- DO NOT explain. DO NOT output text. Only output a numeric score.

Content:
{text}
"""
    for attempt in range(attempts):
        try:
            response = llm.generate_content(
                prompt,
                request_options={"timeout": timeout}   # ⏳ Timeout HERE
            )
            score_str = response.text.strip()
            return float(score_str)
        
        except Exception as e:
            print(f"[Retry {attempt+1}] Error:", e)
            time.sleep(1.5 + random.random()*1.0)

def main():
    # Read csv, filer by content
    df = pd.read_csv('../data/news/TCB_with_content.csv')
    print('Original shape ', df.shape)
    df = df[df['content'].str.len() > 0]
    print('Shape after filtering ', df.shape)
    df['sentiment_score'] = 0.0
    for i in range(df.shape[0]):
        score = sentiment(df.iloc[i]['content'])
        df.at[df.index[i], 'sentiment_score'] = score
        print(f'Completed {i+1} documents.')
        time.sleep(5)

    # Save df 
    df.to_csv("../data/sentiment/TCB_sentiment2.csv", index=False)

if __name__ == "__main__":
    main()