import logging
import asyncio
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
)
from livekit.agents.llm import ChatContext, ChatMessage, StopResponse
from livekit.agents.voice import UserInputTranscribedEvent
from livekit.plugins import cartesia, deepgram, groq, silero

load_dotenv()
logger = logging.getLogger("test")

IGNORE_WORDS = {
    "yeah", "yep", "yes", "ok", "okay", "right",
    "uh-huh", "hmm", "sure", "got it", "mhm", "ah", "oh", "i see",
}


def should_interrupt(text: str) -> bool:
    clean_text = text.strip().lower()
    clean_text = clean_text.replace(".", "").replace("!", "").replace("?", "").replace(",", "")
    
    if not clean_text:
        return False

    words = clean_text.split()
    return any(word not in IGNORE_WORDS for word in words)

class IntelligentInterruptAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice assistant. 
            When asked a question, give a DETAILED and LONG response (at least 3-4 paragraphs).
            Speak naturally and take your time explaining things thoroughly.
            If interrupted, acknowledge it briefly and stop.""",
            stt=deepgram.STT(),
            llm=groq.LLM(), 
            tts=cartesia.TTS(),
        )

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        user_text = (new_message.text_content or "").strip()
        
        if not should_interrupt(user_text):
            logger.info(f"IGNORING TURN (StopResponse): '{user_text}'")
            raise StopResponse()
        
        logger.info(f"ACCEPTING INPUT: '{user_text}'")

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        allow_interruptions=False,
        discard_audio_if_uninterruptible=False, 
    )
    
    agent = IntelligentInterruptAgent()
    
    @session.on("user_input_transcribed")
    def on_transcription_hook(ev: UserInputTranscribedEvent):
        logger.info(f"EVENT: is_final={ev.is_final}, text='{ev.transcript}'")
        
        if not ev.is_final:
            return
        
        is_speaking = session.current_speech is not None
        logger.info(f"FINAL: '{ev.transcript}' | Speaking: {is_speaking}")
        
        if is_speaking:
            if should_interrupt(ev.transcript):
                logger.info(f"MANUAL INTERRUPT: '{ev.transcript}'")
                asyncio.ensure_future(session.interrupt(force=True))  
            else:
                logger.info(f"IGNORING BACKCHANNEL: '{ev.transcript}'")

    logger.info("Event listener registered")
    
    await session.start(agent=agent, room=ctx.room)
    logger.info("Session started - ready for conversation!")
    
    await session.say("Hello! Ask me any question and I will give you a detailed answer. Try saying Yeah while I talk, and then try saying Stop.")


if __name__ == "__main__":
    cli.run_app(server)
