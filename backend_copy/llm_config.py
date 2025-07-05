# llm_config.py
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
import requests

def web_search(query):
    """Combined search across Islamic sources"""
    try:
        results = []
        
        # 1. Search Quran API
        try:
            response = requests.get("http://api.alquran.cloud/v1/surah")
            if response.status_code == 200:
                surahs = response.json()['data']
                query_lower = query.lower()
                
                for surah in surahs:
                    if any(word in surah['englishName'].lower() for word in query_lower.split()):
                        results.append(f"📖 Quran - Surah {surah['englishName']}: {surah['englishNameTranslation']}")
                        break
        except:
            pass
        
        # 2. Search Wikipedia for Islamic content
        try:
            search_response = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/search",
                params={"q": f"{query} Islam", "limit": 2}
            )
            
            if search_response.status_code == 200:
                for page in search_response.json().get('pages', []):
                    title = page['title']
                    summary_response = requests.get(
                        f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
                    )
                    
                    if summary_response.status_code == 200:
                        extract = summary_response.json().get('extract', '')
                        if extract and len(extract) > 50:
                            results.append(f"📚 Wikipedia - {title}: {extract[:150]}...")
        except:
            pass
        
        # 3. Add Islamic resource links
        results.append("🔗 Additional Islamic Resources:")
        results.append("• Sunnah.com - Authentic Hadith Collection")
        results.append("• IslamHouse.com - Islamic Literature")
        
        return results if results else [f"No Islamic resources found for '{query}'"]
    
    except Exception as e:
        return [f"Error in Islamic search: {str(e)}"]

class SharedLLMManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_llm()
            self._setup_chat_engine()
            self._initialized = True
    
    def _setup_llm(self):
        self.llm = HuggingFaceLLM(
            model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
            tokenizer_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
            context_window=1024,
            max_new_tokens=256,
            device_map="auto"
        )
        
        self.embedding_llm = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Set global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embedding_llm
        Settings.chunk_size = 256
    
    def _setup_chat_engine(self):
        # Load index
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        self.index = load_index_from_storage(storage_context=storage_context)
        
        # Setup memory
        chat_store = SimpleChatStore()
        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=256,
            chat_store=chat_store,
            chat_store_key="user1",
        )
        
        # Persist chat store
        chat_store.persist(persist_path="chat_store.json")
        
        # Create chat engine
        self.chat_engine = self.index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=self.memory,
            llm=self.llm,
            context_prompt=(
                "Answer only in German. "
                "You are a German chatbot, able to have normal interactions, as well as talk "
                "about modules, technical information about the modules, and informations from the Technical University of Berlin. "
                "Here are the relevant documents for the context:\n"
                "{context_str}"
                "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
            ),
            verbose=False,
            fallback_handler=web_search,
        )
    
    def get_llm(self):
        """Get direct LLM access"""
        return self.llm
    
    def get_chat_engine(self):
        """Get the full chat engine"""
        return self.chat_engine
    
    def get_embedding_model(self):
        """Get embedding model"""
        return self.embedding_llm

# Convenience functions
def get_llm():
    return SharedLLMManager().get_llm()

def get_chat_engine():
    return SharedLLMManager().get_chat_engine()

def get_embedding_model():
    return SharedLLMManager().get_embedding_model()