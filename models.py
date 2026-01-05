import os
from langchain.chat_models import init_chat_model
import faiss
from transformers import CLIPProcessor, CLIPModel

from dotenv import load_dotenv
load_dotenv() 

GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")

model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_index = faiss.read_index("clip_image_index.faiss")