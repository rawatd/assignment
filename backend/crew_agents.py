from crewai import Agent, Task, Crew, Process
from textwrap import dedent
from llama_index.core import QueryBundle
from llama_index.core.schema import TextNode
from llama_index.core.response_synthesizers import CompactAndRefine
from dotenv import load_dotenv
# Import LLM and RAG components from your backend
from backend.models.llm_client import initialize_ollama_llm
from backend.rag_pipeline import get_rag_query_engine, get_llm_instance

from phoenix.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor
import os
load_dotenv()
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
#os.environ["PHOENIX_API_KEY"] ="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MyJ9.DfQqFBQoaFYUBCuXx1PqBSZedEVAfdYcZLJoVqEcsGQ"

# configure the Phoenix tracer
# Instrument CrewAI
#tracer_provider = register(endpoint="http://localhost:6006/v1/traces")
tracer_provider = register(
  project_name="uae_hr_laws_app", # Default is 'default'
  endpoint="http://127.0.0.1:6006/v1/traces",
  auto_instrument=True # Auto-instrument your app based on installed OI dependencies
)
CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)

class ContextualRAGCrew:
    def __init__(self):
        self.llm = get_llm_instance() # Get the initialized LLM
        self.rag_query_engine = get_rag_query_engine() # Get the initialized LlamaIndex query engine

    def create_agents(self):
        """Defines the CrewAI Agents for the RAG pipeline."""

        # Contextual Retriever Agent
        self.retriever_agent = Agent(
            role='Contextual Document Retriever',
            goal=dedent("""
                Identify and retrieve the most relevant information from the knowledge base 
                based on the user's query and the ongoing conversation history. 
                Focus on providing precise and comprehensive context for answer generation.
                Use the LlamaIndex query engine for retrieval, ensuring re-ranking is applied.
            """),
            backstory=dedent("""
                An expert in efficiently searching and extracting information from diverse
                documents, skilled at understanding user intent and conversational context
                to fetch highly relevant passages.
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=False, # This agent performs retrieval, doesn't delegate.
            # tools=[self.rag_query_engine] # CrewAI tools are functions, not directly query_engine
        )

        # Information Synthesizer Agent
        self.synthesizer_agent = Agent(
            role='Information Synthesizer and Answer Generator',
            goal=dedent("""
                Synthesize information from the retrieved documents and the user's query
                into a clear, concise, and accurate answer. The answer must be directly
                supported by the provided context and adhere to the conversation flow.
            """),
            backstory=dedent("""
                A meticulous and articulate expert in language generation, able to
                transform raw information into coherent, user-friendly responses while
                maintaining factual accuracy and contextual relevance.
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=False # This agent is responsible for final answer generation.
        )
        return self.retriever_agent, self.synthesizer_agent

    def create_tasks(self, user_query: str, chat_history: list = None):
        """Defines the CrewAI Tasks for the RAG pipeline."""
        
        # Combine query and history for contextual retrieval
        # CrewAI agents can be instructed to use chat history in their thinking
        full_contextual_query = f"User Query: {user_query}\nChat History: {chat_history if chat_history else 'No previous conversation.'}"

        # Task 1: Contextual Retrieval
        # This task will leverage the LlamaIndex query engine directly
        # For CrewAI, a "tool" is usually a function. We'll define a simple wrapper.
        def perform_retrieval(query: str):
            """Retrieves relevant documents using the RAG query engine."""
            print(f"Retrieving for query: {query}")
            # LlamaIndex query engine already handles embedding and re-ranking
            # We want to get the raw nodes/text that were retrieved and used
            response_obj = self.rag_query_engine.query(query)
            
            # Extract text from source nodes
            retrieved_texts = [node.text for node in response_obj.source_nodes]
            return {"answer": str(response_obj), "context_nodes": retrieved_texts}

        retrieval_task = Task(
            description=dedent(f"""
                Based on the following query and chat history, use the knowledge base 
                to retrieve the most relevant document chunks.
                Query: '{user_query}'
                Chat History: '{chat_history if chat_history else 'None'}'
                
                Your output should contain the raw text content of the retrieved relevant documents.
            """),
            expected_output="A list of relevant document text snippets that comprehensively address the user's query in context.",
            agent=self.retriever_agent,
            #output_json=True, # Attempt to get structured output
            callback=lambda output: (
                print(f"Retriever Agent Output: {str(output.raw)}"), # Sometimes it's 'raw'
                # Try outputting the whole object if 'raw' also fails
                # print(f"Retriever Agent Output: {str(output)}")
            )
        )
        
        # Task 2: Answer Synthesis
        synthesis_task = Task(
            description=dedent(f"""
                Given the user's query: '{user_query}', and the retrieved context: 
                '{{retrieval_task.output.context_nodes}}'.
                Synthesize a concise, accurate, and comprehensive answer. 
                Ensure the answer directly addresses the user's query using ONLY the provided context. 
                Do not make up information. If the answer cannot be found in the context, state that clearly.
                Consider the following chat history for generating a contextually relevant response:
                Chat History: '{chat_history if chat_history else 'None'}'
            """),
            expected_output="A direct and complete answer to the user's query based solely on the provided context.",
            agent=self.synthesizer_agent,
            context=[retrieval_task], # The output of retrieval_task is the context for this task
             callback=lambda output: print(f"Retriever Agent Output: {str(output)}")
        )
        
        return retrieval_task, synthesis_task

    def run_crew(self, user_query: str, chat_history: list = None):
        """Runs the CrewAI pipeline."""
        retriever_agent, synthesizer_agent = self.create_agents()
        retrieval_task, synthesis_task = self.create_tasks(user_query, chat_history)

        crew = Crew(
            agents=[retriever_agent, synthesizer_agent],
            tasks=[retrieval_task, synthesis_task],
            verbose=True, # Increased verbosity for debugging
            process=Process.sequential # Agents work one after another
        )
        print("Starting CrewAI process...")
        result = crew.kickoff()
        print("CrewAI process finished.")
        return result

if __name__ == '__main__':
    # This block is for testing the CrewAI setup directly
    # Ensure your DB has data from Step 4 and Ollama is running
    print("Testing CrewAI pipeline directly...")
    crew_instance = ContextualRAGCrew()
    
    # Example chat history
    test_chat_history = [
        {"role": "user", "content": "Tell me about the HR policies."},
        {"role": "assistant", "content": "The HR bylaws cover various aspects incl1uding leave policies, code of conduct, and performance management."}
    ]

    #query = "Based on the introduction and scope, what is the primary purpose of the Abu Dhabi Procurement Standards, and how does it aim to achieve it?"
    query = "Summarize the overall objective of Decision No. (10) of 2020 as it relates to human resources in the Emirate of Abu Dhabi.?"
    try:
        # Note: CrewAI agents are designed to *reason* and *use tools*.
        # Our `rag_query_engine` is *already* a full RAG pipeline (retrieve + LLM generation).
        # We need to explicitly instruct the CrewAI agents to use its output effectively.
        # This setup tries to guide the agents through the RAG steps.
        
        result = crew_instance.run_crew(query, test_chat_history)
        #result = crew_instance.run_crew(query)
        print("\nFinal Answer from CrewAI:")
        print(result)

    except Exception as e:
        print(f"Error running CrewAI: {e}")