import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONITORED_ROOT: str = os.path.abspath("d:/SEFS/monitored_root")
    UPLOAD_STORAGE: str = os.path.abspath("d:/SEFS/upload_storage")
    VECTOR_DB_PATH: str = os.path.join(os.path.dirname(__file__), "vectors.db")
    FAISS_INDEX_PATH: str = os.path.join(os.path.dirname(__file__), "faiss_index.bin")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_SIMILARITY: int = 5  # Number of top chunk-pair similarities to average
    USE_GPU: bool = True
    
    # LLM Settings
    CEREBRAS_API_URL: str = "https://api.cerebras.ai/v1/chat/completions"
    CEREBRAS_API_KEYS: list[str] = [
        "csk-52m4dv4chcpf9vy9jcmjrevnp5ft22y2vctd68wyr8dewndw",
        "csk-nnj93n833cr4c9rd2vttjeew3nwv494px62jfy45fmwjdch8",
        "csk-f428j58fvtm3n5vent8mjm3nfnwvk2rrvx6vww95e9wctv6c",
        "csk-c2jjpt5k9kttxd44t9jwyn55vje4m2vmrvdjjkd6h2wphv6m",
        "csk-hhcmv35w3kcvt9nffdyhp5f6m6epre8w3mcx32hwxxmyx85y",
        "csk-8xpfkemny5frctw23ckdp9494vdvy8c344j9vv5f2c5kjdjr",
        "csk-ej64dtyre2k3nex8e3jykxh6cdvr5fxc3vpd8yrnyjxeyk3h",
        "csk-nh93kjmpvmw3dwnn9m23jye2969fjfrwxfcy69dmkdtvjdc4",
        "csk-nej3xm94wrfth8mp35c9ck3wfdhf684rnckettkmjpt5vnpw",
        "csk-w6jkn2v4rrcvffe5en6t9ydjw854c5332cdfmr9mm4d995p3",
        "csk-vx3wj5h8hpm6fcrfh8r5njcfvp6ntd6rdjf6kvy9nfrmjm6n"
    ]
    LLM_MODEL: str = "llama-3.3-70b"
    LLM_TIMEOUT: int = 30
    LLM_MAX_TOKENS: int = 1024
    
    # Clustering Settings
    SIMILARITY_THRESHOLD: float = 0.7
    TOP_K_NEIGHBORS: int = 10
    REBALANCE_INTERVAL: int = 3600  # seconds
    MIN_CLUSTER_SIZE: int = 2
    
    # Caching
    SUMMARY_CACHE_TTL: int = 86400  # 24 hours
    CLUSTER_NAME_CACHE_TTL: int = 604800  # 7 days
    
    # Natural Language Commands (gpt-oss-120b reasoning model)
    NL_API_URL: str = "https://api.cerebras.ai/v1"
    NL_MODEL: str = "gpt-oss-120b"
    NL_API_KEY: str = "csk-52m4dv4chcpf9vy9jcmjrevnp5ft22y2vctd68wyr8dewndw"
    NL_MAX_TOKENS: int = 2048
    NL_TIMEOUT: int = 60
    
    # Duplicate / Cross-cluster thresholds
    DUPLICATE_THRESHOLD: float = 0.92
    CROSS_CLUSTER_MIN: float = 0.45
    CROSS_CLUSTER_MAX: float = 0.70
    
settings = Settings()