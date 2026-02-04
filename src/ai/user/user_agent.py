from ai.agent.agent import Agent
from ai.user.user_config import UserConfig
from ai.agent.agent_factory import AgentFactory

class UserAgent:
    def root(self) -> Agent:
        conf = UserConfig()
        name = conf.rootAgentName()
        factory = AgentFactory( conf )
        return factory.load( name )

