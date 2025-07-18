import streamlit as st
import asyncio
import sys
import os
from typing import Dict, Any, List
import logging

# Add project root to path
sys.path.append('/app')

from agents.agent_orchestrator import AgentOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AutoGen Multi-Agent AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    .agent-selected {
        border-color: #1f77b4;
        background: #e7f3ff;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .agent-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-success { background-color: #4caf50; }
    .status-error { background-color: #f44336; }
    .status-loading { background-color: #ff9800; }
</style>
""", unsafe_allow_html=True)

class AutoGenApp:
    def __init__(self):
        self.orchestrator = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'orchestrator' not in st.session_state:
            st.session_state.orchestrator = None
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'selected_agents' not in st.session_state:
            st.session_state.selected_agents = ["research"]
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
    
    async def initialize_system(self):
        """Initialize the orchestrator and agents"""
        if not st.session_state.system_initialized:
            with st.spinner("🚀 Initializing AI Agents..."):
                orchestrator = AgentOrchestrator()
                success = await orchestrator.initialize_agents()
                
                if success:
                    st.session_state.orchestrator = orchestrator
                    st.session_state.system_initialized = True
                    st.success("✅ All agents initialized successfully!")
                    return True
                else:
                    st.error("❌ Failed to initialize agents")
                    return False
        return True
    
    def render_sidebar(self):
        """Render the sidebar with agent selection and controls"""
        st.sidebar.title("🤖 Agent Control Panel")
        
        # System status
        st.sidebar.subheader("System Status")
        if st.session_state.system_initialized:
            st.sidebar.markdown('<span class="status-indicator status-success"></span>System Online', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<span class="status-indicator status-error"></span>System Offline', unsafe_allow_html=True)
        
        # Agent selection
        st.sidebar.subheader("Select Agents")
        
        agent_descriptions = {
            "research": {
                "name": "🔍 Research Agent",
                "description": "Conducts web research and gathers information",
                "model": "Phi-3 Mini"
            },
            "documentation": {
                "name": "📝 Documentation Agent", 
                "description": "Creates structured documentation and reports",
                "model": "Mistral 7B"
            },
            "coding": {
                "name": "💻 Coding Agent",
                "description": "Generates, reviews, and debugs code",
                "model": "Qwen 7B"
            }
        }
        
        selected_agents = []
        for agent_key, agent_info in agent_descriptions.items():
            col1, col2 = st.sidebar.columns([1, 3])
            
            with col1:
                is_selected = st.checkbox("", value=agent_key in st.session_state.selected_agents, key=f"agent_{agent_key}")
            
            with col2:
                st.markdown(f"""
                <div class="agent-card {'agent-selected' if is_selected else ''}">
                    <strong>{agent_info['name']}</strong><br>
                    <small>{agent_info['description']}</small><br>
                    <em>Model: {agent_info['model']}</em>
                </div>
                """, unsafe_allow_html=True)
            
            if is_selected:
                selected_agents.append(agent_key)
        
        st.session_state.selected_agents = selected_agents
        
        # System controls
        st.sidebar.subheader("System Controls")
        
        if st.sidebar.button("🔄 Reinitialize System"):
            st.session_state.system_initialized = False
            st.session_state.orchestrator = None
            st.experimental_rerun()
        
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()
        
        # System information
        if st.session_state.orchestrator:
            st.sidebar.subheader("System Information")
            status = st.session_state.orchestrator.get_agent_status()
            
            st.sidebar.write("**Loaded Models:**")
            for model in status["model_info"]["loaded_models"]:
                st.sidebar.write(f"• {model}")
            
            st.sidebar.write(f"**Conversations:** {status['conversation_count']}")
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.markdown('<h1 class="main-header">🤖 AutoGen Multi-Agent AI Assistant</h1>', unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            self.render_message(message)
        
        # Chat input
        if prompt := st.chat_input("Ask me anything! I'll use the selected agents to help you."):
            self.handle_user_input(prompt)
    
    def render_message(self, message: Dict[str, Any]):
        """Render a single message in the chat"""
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
        
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(f'<div class="chat-message agent-message">{message["content"]}</div>', unsafe_allow_html=True)
                
                # Display detailed results if available
                if "results" in message:
                    self.render_agent_results(message["results"])
    
    def render_agent_results(self, results: Dict[str, Any]):
        """Render detailed results from each agent"""
        for agent_name, result in results.items():
            if result.get("status") == "completed":
                with st.expander(f"📋 {agent_name.title()} Agent Results"):
                    
                    if agent_name == "research":
                        st.subheader("Research Report")
                        st.markdown(result.get("research_report", ""))
                        
                        if "sources" in result:
                            st.subheader("Sources")
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(f"{i}. [{source['title']}]({source['url']})")
                    
                    elif agent_name == "documentation":
                        st.subheader("Documentation")
                        st.markdown(result.get("documentation", ""))
                    
                    elif agent_name == "coding":
                        st.subheader("Generated Code")
                        if "code_blocks" in result:
                            for block in result["code_blocks"]:
                                st.code(block["code"], language=block["language"])
                        
                        if "explanation" in result:
                            st.subheader("Explanation")
                            st.markdown(result["explanation"])
            else:
                st.error(f"❌ {agent_name.title()} Agent failed: {result.get('error', 'Unknown error')}")
    
    def handle_user_input(self, prompt: str):
        """Handle user input and process with selected agents"""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Check if system is initialized
        if not st.session_state.system_initialized:
            st.error("Please wait for the system to initialize.")
            return
        
        # Check if agents are selected
        if not st.session_state.selected_agents:
            st.warning("Please select at least one agent.")
            return
        
        # Process request
        with st.spinner(f"Processing with {', '.join(st.session_state.selected_agents)} agents..."):
            try:
                # Run async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    st.session_state.orchestrator.process_user_request(
                        prompt, st.session_state.selected_agents
                    )
                )
                
                # Create response message
                response_content = f"I've processed your request using the {', '.join(st.session_state.selected_agents)} agents. Here are the results:"
                
                assistant_message = {
                    "role": "assistant",
                    "content": response_content,
                    "results": result.get("results", {})
                }
                
                st.session_state.messages.append(assistant_message)
                
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
            
            finally:
                loop.close()
        
        st.experimental_rerun()
    
    def run(self):
        """Main application entry point"""
        # Initialize system
        if not st.session_state.system_initialized:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(self.initialize_system())
            loop.close()
            
            if not success:
                st.stop()
        
        # Render UI
        self.render_sidebar()
        self.render_chat_interface()

def main():
    """Main function"""
    app = AutoGenApp()
    app.run()

if __name__ == "__main__":
    main()

