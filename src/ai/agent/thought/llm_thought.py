from ai.agent.thought.thought import Thought

from ai.agent.thought.conclusion import Conclusion

class LLMThought(Thought):
    def __init__(self, action: str, prompt, llm):
        super().__init__(action)
        self._llm = llm
        self._prompt = prompt
    
    def resolve(self) -> Conclusion:
        content = self._llm.invoke( self._prompt ).content
        return Conclusion( content )
