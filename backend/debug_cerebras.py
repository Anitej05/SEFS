
import asyncio
import aiohttp
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config from current setup
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
# Using the first key for testing
API_KEY = "csk-52m4dv4chcpf9vy9jcmjrevnp5ft22y2vctd68wyr8dewndw" 
MODEL = "llama-3.3-70b"

async def test_api():
    logger.info(f"Testing Cerebras API connection to {CEREBRAS_API_URL}")
    
    prompt = "Why is fast inference important?"
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.3
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    logger.info(f"Sending request with header: Authorization: Bearer {API_KEY[:4]}...{API_KEY[-4:]}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                CEREBRAS_API_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                status = response.status
                logger.info(f"Response Status: {status}")
                
                body = await response.text()
                logger.info(f"Raw Response Body: {body}")
                
                if status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    logger.info(f"Success! Model content: {content}")
                else:
                    logger.error("Request failed.")

    except Exception as e:
        import traceback
        logger.error(f"Exception during request: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_api())
