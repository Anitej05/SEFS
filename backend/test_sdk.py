
import asyncio
import os
import logging
from cerebras.cloud.sdk import AsyncCerebras

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = "csk-52m4dv4chcpf9vy9jcmjrevnp5ft22y2vctd68wyr8dewndw"
MODEL = "llama-3.3-70b"

async def test_sdk():
    logger.info("Testing AsyncCerebras client...")
    
    try:
        client = AsyncCerebras(api_key=API_KEY)
        
        response = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "List 3 benefits of fast inference.\n1.",
                }
            ],
            model=MODEL,
            temperature=0.3,
            max_tokens=200
        )
        
        content = response.choices[0].message.content
        logger.info(f"Success! Content:\n{content}")
        
    except Exception as e:
        logger.error(f"SDK Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_sdk())
