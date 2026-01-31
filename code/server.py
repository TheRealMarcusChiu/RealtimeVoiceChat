from queue import Empty
import logging
from logsetup import setup_logging
from upsample_overlap import UpsampleOverlap
from datetime import datetime
from colors import Colors
import uvicorn
import asyncio
import struct
import json
import time
import threading
import re
from typing import Any, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, Response
from audio_in import AudioInputProcessor
from speech_pipeline_manager import SpeechPipelineManager

setup_logging(logging.INFO)
LOGGER = logging.getLogger(__name__)

TTS_START_ENGINE = "kokoro"
TTS_ORPHEUS_MODEL = "orpheus-3b-0.1-ft-Q8_0-GGUF/orpheus-3b-0.1-ft-q8_0.gguf"
LLM_START_PROVIDER = "ollama"
LLM_START_MODEL = "qwen2.5:3b"
NO_THINK = False
MAX_AUDIO_QUEUE_SIZE = 50
LANGUAGE = "en"
TTS_FINAL_TIMEOUT = 1.0


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Dict[str, Any]) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers.pop("etag", None)
        response.headers.pop("last-modified", None)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("🖥️▶️ Server starting up")
    app.state.SpeechPipelineManager = SpeechPipelineManager(
        tts_engine=TTS_START_ENGINE, llm_provider=LLM_START_PROVIDER,
        llm_model=LLM_START_MODEL, no_think=NO_THINK, orpheus_model=TTS_ORPHEUS_MODEL)
    app.state.Upsampler = UpsampleOverlap()
    app.state.AudioInputProcessor = AudioInputProcessor(
        LANGUAGE, is_orpheus=TTS_START_ENGINE=="orpheus",
        pipeline_latency=app.state.SpeechPipelineManager.full_output_pipeline_latency / 1000)
    app.state.Aborting = False
    yield
    LOGGER.info("🖥️⏹️ Server shutting down")
    app.state.AudioInputProcessor.shutdown()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, 
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", NoCacheStaticFiles(directory="static"), name="static")


@app.get("/")
async def get_index() -> HTMLResponse:
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


def parse_json_message(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        LOGGER.warning("🖥️⚠️ Ignoring client message with invalid JSON")
        return {}


def format_timestamp_ns(timestamp_ns: int) -> str:
    seconds, remainder_ns = divmod(timestamp_ns, 1_000_000_000)
    dt = datetime.fromtimestamp(seconds)
    milliseconds = remainder_ns // 1_000_000
    return f"{dt.strftime('%H:%M:%S')}.{milliseconds:03d}"


async def process_incoming_data(ws: WebSocket, app: FastAPI, incoming_chunks: asyncio.Queue, 
                                callbacks: 'TranscriptionCallbacks') -> None:
    try:
        while True:
            msg = await ws.receive()
            
            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]
                if len(raw) < 8:
                    LOGGER.warning("🖥️⚠️ Received packet too short for 8‑byte header.")
                    continue

                timestamp_ms, flags = struct.unpack("!II", raw[:8])
                client_sent_ns = timestamp_ms * 1_000_000
                server_ns = time.time_ns()

                metadata = {
                    "client_sent_ms": timestamp_ms, "client_sent": client_sent_ns,
                    "client_sent_formatted": format_timestamp_ns(client_sent_ns),
                    "isTTSPlaying": bool(flags & 1), "pcm": raw[8:],
                    "server_received": server_ns,
                    "server_received_formatted": format_timestamp_ns(server_ns)
                }

                if incoming_chunks.qsize() < MAX_AUDIO_QUEUE_SIZE:
                    await incoming_chunks.put(metadata)
                else:
                    LOGGER.warning(f"🖥️⚠️ Audio queue full ({incoming_chunks.qsize()}/{MAX_AUDIO_QUEUE_SIZE}); dropping chunk.")

            elif "text" in msg and msg["text"]:
                data = parse_json_message(msg["text"])
                msg_type = data.get("type")
                LOGGER.info(Colors.apply(f"🖥️📥 ←←Client: {data}").orange)

                if msg_type == "tts_start":
                    LOGGER.info("🖥️ℹ️ Received tts_start from client.")
                    callbacks.tts_client_playing = True
                elif msg_type == "tts_stop":
                    LOGGER.info("🖥️ℹ️ Received tts_stop from client.")
                    callbacks.tts_client_playing = False
                elif msg_type == "clear_history":
                    LOGGER.info("🖥️ℹ️ Received clear_history from client.")
                    app.state.SpeechPipelineManager.reset()
                elif msg_type == "set_speed":
                    speed_factor = data.get("speed", 0) / 100.0
                    turn_detection = app.state.AudioInputProcessor.transcriber.turn_detection
                    if turn_detection:
                        turn_detection.update_settings(speed_factor)
                        LOGGER.info(f"🖥️⚙️ Updated turn detection settings to factor: {speed_factor:.2f}")
    
    except (asyncio.CancelledError, WebSocketDisconnect, RuntimeError) as e:
        if not isinstance(e, asyncio.CancelledError):
            LOGGER.warning(f"🖥️⚠️ Error in process_incoming_data: {repr(e)}")
    except Exception as e:
        LOGGER.exception(f"🖥️💥 Exception in process_incoming_data: {repr(e)}")


async def send_text_messages(ws: WebSocket, message_queue: asyncio.Queue) -> None:
    try:
        while True:
            await asyncio.sleep(0.001)
            data = await message_queue.get()
            if data.get("type") != "tts_chunk":
                LOGGER.info(Colors.apply(f"🖥️📤 →→Client: {data}").orange)
            await ws.send_json(data)
    except (asyncio.CancelledError, WebSocketDisconnect, RuntimeError) as e:
        if not isinstance(e, asyncio.CancelledError):
            LOGGER.warning(f"🖥️⚠️ Error in send_text_messages: {repr(e)}")
    except Exception as e:
        LOGGER.exception(f"🖥️💥 Exception in send_text_messages: {repr(e)}")


async def _reset_interrupt_flag_async(app: FastAPI, callbacks: 'TranscriptionCallbacks'):
    await asyncio.sleep(1)
    if app.state.AudioInputProcessor.interrupted:
        LOGGER.info(f"{Colors.apply('🖥️🎙️ ▶️ Microphone continued (async reset)').cyan}")
        app.state.AudioInputProcessor.interrupted = False
        callbacks.interruption_time = 0
        LOGGER.info(Colors.apply("🖥️🎙️ interruption flag reset after TTS chunk (async)").cyan)


async def send_tts_chunks(app: FastAPI, message_queue: asyncio.Queue, 
                         callbacks: 'TranscriptionCallbacks') -> None:
    try:
        LOGGER.info("🖥️🔊 Starting TTS chunk sender")
        last_quick_answer_chunk = last_chunk_sent = 0
        prev_status = None

        while True:
            await asyncio.sleep(0.001)

            if (app.state.AudioInputProcessor.interrupted and callbacks.interruption_time 
                and time.time() - callbacks.interruption_time > 2.0):
                app.state.AudioInputProcessor.interrupted = False
                callbacks.interruption_time = 0
                LOGGER.info(Colors.apply("🖥️🎙️ interruption flag reset after 2 seconds").cyan)

            mgr = app.state.SpeechPipelineManager
            is_tts_finished = mgr.is_valid_gen() and mgr.running_generation.audio_quick_finished

            curr_status = (int(callbacks.tts_to_client), int(callbacks.tts_client_playing),
                          int(callbacks.tts_chunk_sent), 1, int(callbacks.is_hot),
                          int(callbacks.synthesis_started), int(mgr.running_generation is not None),
                          int(mgr.is_valid_gen()), int(is_tts_finished),
                          int(app.state.AudioInputProcessor.interrupted))

            if curr_status != prev_status:
                LOGGER.info(f"{Colors.apply('🖥️🚦 State ').red} ToClient {curr_status[0]}, "
                           f"ttsClientON {curr_status[1]}, ChunkSent {curr_status[2]}, "
                           f"hot {curr_status[4]}, synth {curr_status[5]} gen {curr_status[6]} "
                           f"valid {curr_status[7]} tts_q_fin {curr_status[8]} mic_inter {curr_status[9]}")
                prev_status = curr_status

            if (not callbacks.tts_to_client or not mgr.running_generation 
                or mgr.running_generation.abortion_started):
                continue

            if not mgr.running_generation.audio_quick_finished:
                mgr.running_generation.tts_quick_allowed_event.set()

            if not mgr.running_generation.quick_answer_first_chunk_ready:
                continue

            chunk = None
            try:
                chunk = mgr.running_generation.audio_chunks.get_nowait()
                if chunk:
                    last_quick_answer_chunk = time.time()
            except Empty:
                if (not mgr.running_generation.quick_answer_provided 
                    or mgr.running_generation.audio_final_finished):
                    LOGGER.info("🖥️🏁 Sending of TTS chunks and 'user request/assistant answer' cycle finished.")
                    callbacks.send_final_assistant_answer()
                    mgr.running_generation = None
                    callbacks.tts_chunk_sent = False
                    callbacks.reset_state()
                continue

            base64_chunk = app.state.Upsampler.get_base64_chunk(chunk)
            message_queue.put_nowait({"type": "tts_chunk", "content": base64_chunk})
            last_chunk_sent = time.time()

            if not callbacks.tts_chunk_sent:
                asyncio.create_task(_reset_interrupt_flag_async(app, callbacks))
            callbacks.tts_chunk_sent = True

    except (asyncio.CancelledError, WebSocketDisconnect, RuntimeError) as e:
        if not isinstance(e, asyncio.CancelledError):
            LOGGER.warning(f"🖥️⚠️ Error in send_tts_chunks: {repr(e)}")
    except Exception as e:
        LOGGER.exception(f"🖥️💥 Exception in send_tts_chunks: {repr(e)}")


class TranscriptionCallbacks:
    def __init__(self, app: FastAPI, message_queue: asyncio.Queue):
        self.app = app
        self.message_queue = message_queue
        self.final_transcription = self.abort_text = self.last_abort_text = ""
        self.tts_to_client = self.user_interrupted = self.tts_chunk_sent = False
        self.tts_client_playing = self.silence_active = False
        self.interruption_time = 0.0
        self.is_hot = self.user_finished_turn = self.synthesis_started = False
        self.assistant_answer = self.final_assistant_answer = ""
        self.is_processing_potential = self.is_processing_final = False
        self.last_inferred_transcription = self.partial_transcription = ""
        self.final_assistant_answer_sent = False
        self.reset_state()
        self.abort_request_event = threading.Event()
        self.abort_worker_thread = threading.Thread(target=self._abort_worker, daemon=True)
        self.abort_worker_thread.start()

    def reset_state(self):
        self.tts_to_client = self.user_interrupted = self.tts_chunk_sent = False
        self.interruption_time = 0.0
        self.silence_active = True
        self.is_hot = self.user_finished_turn = self.synthesis_started = False
        self.assistant_answer = self.final_assistant_answer = ""
        self.is_processing_potential = self.is_processing_final = False
        self.last_inferred_transcription = self.partial_transcription = ""
        self.final_assistant_answer_sent = False
        self.app.state.AudioInputProcessor.abort_generation()

    def _abort_worker(self):
        while True:
            if self.abort_request_event.wait(timeout=0.1):
                self.abort_request_event.clear()
                if self.last_abort_text != self.abort_text:
                    self.last_abort_text = self.abort_text
                    LOGGER.debug(f"🖥️🧠 Abort check triggered by partial: '{self.abort_text}'")
                    self.app.state.SpeechPipelineManager.check_abort(self.abort_text, False, "on_partial")

    def on_partial(self, txt: str):
        self.final_assistant_answer_sent = False
        self.final_transcription = ""
        self.partial_transcription = txt
        self.message_queue.put_nowait({"type": "partial_user_request", "content": txt})
        self.abort_text = txt
        self.abort_request_event.set()

    def safe_abort_running_syntheses(self, reason: str):
        pass

    def on_tts_allowed_to_synthesize(self):
        gen = self.app.state.SpeechPipelineManager.running_generation
        if gen and not gen.abortion_started:
            LOGGER.info(f"{Colors.apply('🖥️🔊 TTS ALLOWED').blue}")
            gen.tts_quick_allowed_event.set()

    def on_potential_sentence(self, txt: str):
        LOGGER.debug(f"🖥️🧠 Potential sentence: '{txt}'")
        self.app.state.SpeechPipelineManager.prepare_generation(txt)

    def on_potential_final(self, txt: str):
        LOGGER.info(f"{Colors.apply('🖥️🧠 HOT: ').magenta}{txt}")

    def on_potential_abort(self):
        pass

    def on_before_final(self, audio: bytes, txt: str):
        LOGGER.info(Colors.apply('🖥️🏁 =================== USER TURN END ===================').light_gray)
        self.user_finished_turn = True
        self.user_interrupted = False
        
        mgr = self.app.state.SpeechPipelineManager
        if mgr.is_valid_gen():
            LOGGER.info(f"{Colors.apply('🖥️🔊 TTS ALLOWED (before final)').blue}")
            mgr.running_generation.tts_quick_allowed_event.set()

        if not self.app.state.AudioInputProcessor.interrupted:
            LOGGER.info(f"{Colors.apply('🖥️🎙️ ⏸️ Microphone interrupted (end of turn)').cyan}")
            self.app.state.AudioInputProcessor.interrupted = True
            self.interruption_time = time.time()

        LOGGER.info(f"{Colors.apply('🖥️🔊 TTS STREAM RELEASED').blue}")
        self.tts_to_client = True

        user_request_content = self.final_transcription or self.partial_transcription
        self.message_queue.put_nowait({"type": "final_user_request", "content": user_request_content})

        if mgr.is_valid_gen() and mgr.running_generation.quick_answer and not self.user_interrupted:
            self.assistant_answer = mgr.running_generation.quick_answer
            self.message_queue.put_nowait({"type": "partial_assistant_answer", "content": self.assistant_answer})

        LOGGER.info(f"🖥️🧠 Adding user request to history: '{user_request_content}'")
        mgr.history.append({"role": "user", "content": user_request_content})

    def on_final(self, txt: str):
        LOGGER.info(f"\n{Colors.apply('🖥️✅ FINAL USER REQUEST (STT Callback): ').green}{txt}")
        if not self.final_transcription:
            self.final_transcription = txt

    def abort_generations(self, reason: str):
        LOGGER.info(f"{Colors.apply('🖥️🛑 Aborting generation:').blue} {reason}")
        self.app.state.SpeechPipelineManager.abort_generation(reason=f"server.py abort_generations: {reason}")

    def on_silence_active(self, silence_active: bool):
        self.silence_active = silence_active

    def on_partial_assistant_text(self, txt: str):
        LOGGER.info(f"{Colors.apply('🖥️💬 PARTIAL ASSISTANT ANSWER: ').green}{txt}")
        if not self.user_interrupted:
            self.assistant_answer = txt
            if self.tts_to_client:
                self.message_queue.put_nowait({"type": "partial_assistant_answer", "content": txt})

    def on_recording_start(self):
        LOGGER.info(f"{Colors.ORANGE}🖥️🎙️ Recording started.{Colors.RESET} TTS Client Playing: {self.tts_client_playing}")
        if self.tts_client_playing:
            self.tts_to_client = False
            self.user_interrupted = True
            LOGGER.info(f"{Colors.apply('🖥️❗ INTERRUPTING TTS due to recording start').blue}")
            LOGGER.info(Colors.apply("🖥️✅ Sending final assistant answer (forced on interruption)").pink)
            self.send_final_assistant_answer(forced=True)
            self.tts_chunk_sent = False
            LOGGER.info("🖥️🛑 Sending stop_tts to client.")
            self.message_queue.put_nowait({"type": "stop_tts", "content": ""})
            LOGGER.info(f"{Colors.apply('🖥️🛑 RECORDING START ABORTING GENERATION').red}")
            self.abort_generations("on_recording_start, user interrupts, TTS Playing")
            LOGGER.info("🖥️❗ Sending tts_interruption to client.")
            self.message_queue.put_nowait({"type": "tts_interruption", "content": ""})

    def send_final_assistant_answer(self, forced=False):
        final_answer = ""
        mgr = self.app.state.SpeechPipelineManager
        if mgr.is_valid_gen():
            final_answer = mgr.running_generation.quick_answer + mgr.running_generation.final_answer

        if not final_answer:
            if forced and self.assistant_answer:
                final_answer = self.assistant_answer
                LOGGER.warning(f"🖥️⚠️ Using partial answer as final (forced): '{final_answer}'")
            else:
                LOGGER.warning(f"🖥️⚠️ Final assistant answer was empty, not sending.")
                return

        LOGGER.debug(f"🖥️✅ Attempting to send final answer: '{final_answer}' (Sent previously: {self.final_assistant_answer_sent})")

        if not self.final_assistant_answer_sent and final_answer:
            cleaned_answer = re.sub(r'[\r\n]+', ' ', final_answer)
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip().replace('\\n', ' ')
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()

            if cleaned_answer:
                LOGGER.info(f"\n{Colors.apply('🖥️✅ FINAL ASSISTANT ANSWER (Sending): ').green}{cleaned_answer}")
                self.message_queue.put_nowait({"type": "final_assistant_answer", "content": cleaned_answer})
                app.state.SpeechPipelineManager.history.append({"role": "assistant", "content": cleaned_answer})
                self.final_assistant_answer_sent = True
                self.final_assistant_answer = cleaned_answer
            else:
                LOGGER.warning(f"🖥️⚠️ {Colors.YELLOW}Final assistant answer was empty after cleaning.{Colors.RESET}")
                self.final_assistant_answer_sent = False
                self.final_assistant_answer = ""
        elif forced and not final_answer:
            LOGGER.warning(f"🖥️⚠️ {Colors.YELLOW}Forced send of final assistant answer, but it was empty.{Colors.RESET}")
            self.final_assistant_answer = ""


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    LOGGER.info("🖥️✅ Client connected via WebSocket.")

    message_queue = asyncio.Queue()
    audio_chunks = asyncio.Queue()
    callbacks = TranscriptionCallbacks(app, message_queue)

    aip = app.state.AudioInputProcessor
    aip.realtime_callback = callbacks.on_partial
    aip.transcriber.potential_sentence_end = callbacks.on_potential_sentence
    aip.transcriber.on_tts_allowed_to_synthesize = callbacks.on_tts_allowed_to_synthesize
    aip.transcriber.potential_full_transcription_callback = callbacks.on_potential_final
    aip.transcriber.potential_full_transcription_abort_callback = callbacks.on_potential_abort
    aip.transcriber.full_transcription_callback = callbacks.on_final
    aip.transcriber.before_final_sentence = callbacks.on_before_final
    aip.recording_start_callback = callbacks.on_recording_start
    aip.silence_active_callback = callbacks.on_silence_active
    app.state.SpeechPipelineManager.on_partial_assistant_text = callbacks.on_partial_assistant_text

    tasks = [
        asyncio.create_task(process_incoming_data(ws, app, audio_chunks, callbacks)),
        asyncio.create_task(aip.process_chunk_queue(audio_chunks)),
        asyncio.create_task(send_text_messages(ws, message_queue)),
        asyncio.create_task(send_tts_chunks(app, message_queue, callbacks)),
    ]

    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception as e:
        LOGGER.error(f"🖥️💥 {Colors.apply('ERROR').red} in WebSocket session: {repr(e)}")
    finally:
        LOGGER.info("🖥️🧹 Cleaning up WebSocket tasks...")
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        LOGGER.info("🖥️❌ WebSocket session ended.")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_config=None)