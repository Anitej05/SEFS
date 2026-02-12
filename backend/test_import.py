print("Step 1: Starting...")
print("Step 2: Importing sentence_transformers...")
import sentence_transformers
print("Step 3: sentence_transformers imported!")
print("Step 4: Creating model...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Step 5: Model loaded!")
print("Step 6: Testing encoding...")
vec = model.encode(["test"])
print("Step 7: Success! Vector shape:", vec.shape)
