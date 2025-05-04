import os, torch
from glob import glob

from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig
)

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
    Settings
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.agent import ReActAgent
from llama_index.core.postprocessor import SimilarityPostprocessor

os.environ['TRANSFORMERS_CACHE'] = './cache/'
os.environ['HF_HOME'] = './cache'

hf_token= "YOUR HUGGINFACE ACCESS TOKEN GOES HERE"

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=hf_token,
)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

llm = HuggingFaceLLM(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={
        "token": hf_token,
        # "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
        "quantization_config": quantization_config
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
    },
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer_kwargs={"token": hf_token},
    stopping_ids=stopping_ids,
)

# bge embedding model
Settings.embed_model = embed_model

# Llama-3-8B-Instruct model
Settings.llm = llm

msg =  '''YOUR INPUT GOES HERE
        '''
# a vanilla Llama3 agent, not using any RAG databases to supplement.
naive_agent = ReActAgent.from_tools(
    [],
    llm=llm,
    verbose=True,
    max_iterations=100
)

response = naive_agent.chat(msg)

# read in some anti-trans posts
anti_trans_docs = SimpleDirectoryReader(
    input_files=list(glob("anti_trans/*.txt"))
).load_data()

# read in some pro-trans posts
pro_trans_docs = SimpleDirectoryReader(
    input_files=list(glob("pro_trans/*.txt"))
).load_data()

# read in some neutral posts
neu_docs = SimpleDirectoryReader(
    input_files=list(glob("neutral/*.txt"))
).load_data()

# read in our theory codebook
theory_docs = SimpleDirectoryReader(
    input_files=list(glob("transtheory/*"))
).load_data()

# index posts / codebook sentences 
anti_trans_index = VectorStoreIndex.from_documents(anti_trans_docs)
pro_trans_index = VectorStoreIndex.from_documents(pro_trans_docs)
neu_index = VectorStoreIndex.from_documents(neu_docs)
theory_index = VectorStoreIndex.from_documents(theory_docs)

# build a retriever that grabs top 3 matching codebook sentences based on cosine similarity
# can also be done for the other document indices
theory_retriever = VectorIndexRetriever(
    index=theory_index,
    similarity_top_k=3,
)
retrieval_engine = RetrieverQueryEngine.from_args(
    theory_retriever, response_mode="no_text", node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.75)]
)
response = retrieval_engine.query(msg)
print("SOURCES FOUND: ")
for nodescore in response.source_nodes:
    print("document: "+str(nodescore.node.metadata["file_path"]))
    print("text: "+str(nodescore.node.text))
    print("score: "+str(nodescore.score))


# create query engines for llm to pull indexed documents as context to answer classification prompt  
anti_trans_engine = anti_trans_index.as_query_engine(similarity_top_k=3, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.75)])
pro_trans_engine = pro_trans_index.as_query_engine(similarity_top_k=3, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.75)])
neu_engine = neu_index.as_query_engine(similarity_top_k=3, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.75)])
theory_engine = theory_index.as_query_engine(similarity_top_k=3, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.75)])

query_engine_tools = [
    QueryEngineTool(
        query_engine=anti_trans_engine,
        metadata=ToolMetadata(
            name="anti_trans",
            description=(
                "A set of anti-trans TikTok transcripts and descriptions."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=pro_trans_engine,
        metadata=ToolMetadata(
            name="pro_trans",
            description=(
                "A set of pro-trans TikTok transcripts and descriptions."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=neu_engine,
        metadata=ToolMetadata(
            name="neutral",
            description=(
                "A set of neutral TikTok transcripts and descriptions irrelevant to transgender sentiment."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=theory_engine,
        metadata=ToolMetadata(
            name="theory",
            description=(
                "A set of documents outlining aspects of anti-trans hate and how to code trans related sentiment."
            ),
        ),
    ),
]

RAG_agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    max_iterations=100
)


response = RAG_agent.chat(msg)


