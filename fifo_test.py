import os
import pathlib
import subprocess
import sys
import re
from contextlib import contextmanager
import time
# from nltk.tokenize import sent_tokenize
# 
import simpleaudio as sa
import sounddevice as sd
import numpy as np
import pasimple
from TTS.tts.utils.synthesis import synthesis
from nltk.data import load
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor


channels = 1  # Mono audio

@contextmanager
def specific_error(message):
    try:
        yield
    except ValueError as e:
        raise ValueError(message) from e


def DecodeMessage(remainder):
    with specific_error("Did not find separator"):
        header, content = remainder.split('\r\n\r\n', 1)

    with specific_error("Content-Lenght was not an int"):
        content_length = int(header[len("Content-Length: "):])

    if len(content) < content_length:
       raise ValueError(f"Content-Length mismatch: {len(content)=}, {content_length=}")
    
    body = content[:content_length]
    return json.loads(body, strict=False), content[content_length:]

def write_vtt(audio_transcripts, bufname):
    bytes_per_sample = 4
    with open(f"../Subtitles/{bufname.with_suffix('.vtt')}", "w") as vtt_file:
        vtt_file.write("WEBVTT\n\n")
        start_time = 0.0  # Start time in seconds
        for audio_bytes, transcript in audio_transcripts:
            num_samples = len(audio_bytes) / (bytes_per_sample * channels)
            duration = num_samples / sample_rate
            end_time = start_time + duration
            # Format start and end time into h:mm:ss.mmm
            start_time_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{start_time % 60:06.3f}"
            end_time_str = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{end_time % 60:06.3f}"
            vtt_file.write(f"{start_time_str} --> {end_time_str}\n")
            vtt_file.write(transcript + "\n\n")
            start_time = end_time + 0.6


def write_to_file(audio_transcripts : list[tuple[bytes, str]], sample_rate, bufname):
    write_vtt(audio_transcripts, bufname)
    
    command = [
        'ffmpeg', '-y',
        '-f', 'f32le', 
        '-ar', str(sample_rate),  # sample rate
        '-ac', '1',  # number of audio channels
        '-i', '-',  # The input comes from stdin
        '-acodec', 'aac',  # audio codec for M4A
        str(bufname.with_suffix('.m4b'))
    ]
    sentance_pause_duration_seconds = 0.6
    num_samples = int(sample_rate * sentance_pause_duration_seconds )
    bytes_per_sample = 4
    total_bytes = num_samples * bytes_per_sample
    sentance_pause = bytes(total_bytes)

    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    process.stdin.write(sentance_pause.join(wav for wav, _ in audio_transcripts))
    process.stdin.close()
    process.wait()


# text = """\
# hello  world . 



# How are you.
# """
# tokenizer = load(f"tokenizers/punkt/english.pickle")
# spans = list(tokenizer.span_tokenize(text))
# sentances = [text[s:e] for s, e in spans]
# print(sentances)
# exit(0)

cache = {}
load_tts = True
if load_tts:
    from TTS.api import TTS
    tts = TTS(model_name="tts_models/en/vctk/vits", gpu=True)
    sample_rate = tts.synthesizer.output_sample_rate
else:
    sample_rate = 44100  # samples per second
    duration = 2.0  # seconds
    frequency = 440.0  # Hz (A4 note)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False,  dtype=np.float32)
    sine_wave = np.sin(2 * np.pi * frequency * t)

# tts.tts_to_file("Hello world", 'p307')
# wav = tts.tts("Hello world", 'p307')


# outputs = synthesis(
#                     model=tts.synthesizer.tts_model,
#                     text="Hello world",
#                     CONFIG=tts.synthesizer.tts_config,
#                     use_cuda=False,
#                     speaker_id=77, # p307
#                     use_griffin_lim=tts.synthesizer.vocoder_model is None,
#                     )
# sine_wave2 = outputs["wav"]

#sd.play(wav, samplerate=sr)
#sd.wait()

# note = np.array(wav)
# audio = note * (2**15 - 1) / np.max(np.abs(note))
# audio = audio.astype(np.int16)


format = pasimple.PA_SAMPLE_FLOAT32LE  # Little-endian 16-bit
# format = pasimple.PA_SAMPLE_S16LE # Little-endian 16-bit
audio_server = pasimple.PaSimple(pasimple.PA_STREAM_PLAYBACK, format, channels, sample_rate)
# audio_server.write(sine_wave2.tobytes())
# audio_server.drain()
# bytes_ = sine_wave.tobytes()
# audio_server.write(bytes_)
# time.sleep(2)
# exit(0)

# with pasimple.PaSimple(pasimple.PA_STREAM_PLAYBACK, format, channels, sample_rate) as audio_server:
#     audio_server.write(audio.tobytes())
#     audio_server.drain()

# play_obj = sa.play_buffer(audio, 1, 2, sr)
# play_obj.wait_done()
b = 0

def get_audio(sentance):
    print("get_audio")
    if not sentance in cache:
        # print(f"computing: {sentance}")
        wav = synthesis(
            model=tts.synthesizer.tts_model,
            text=sentance,
            CONFIG=tts.synthesizer.tts_config,
            use_cuda=True,
            speaker_id=77, # p307
            use_griffin_lim=tts.synthesizer.vocoder_model is None,
            )["wav"]
        audio = wav.tobytes()
        cache[sentance] = audio
        return audio
    else:
        # print(f"using cached: {sentance}")
        return cache[sentance]

async def get_audio_async(sentance):
    return get_audio(sentance)

async def play_audio(sentances):
    sentance_pause_duration_seconds = 0.6
    num_samples = int(sample_rate * sentance_pause_duration_seconds )
    bytes_per_sample = 4
    total_bytes = num_samples * bytes_per_sample
    sentance_pause = bytes(total_bytes)
    try:
        # # print(f"play_audio. {sentances=}")
        # audio = await asyncio.shield(get_audio_async(sentances[0]))
        # print(f"writing to pulse audio: {sentances[0]=}")
        # audio_server.write(audio)
        # audio_server.write(sentance_pause)

        for sentance in sentances:
            print("starting get audio")
            # if asyncio.current_task().cancelling(): raise asyncio.CancelledError
            await asyncio.sleep(0)
            audio = await asyncio.shield(get_audio_async(sentance))
            print("finishing get audio")
            # if asyncio.current_task().cancelling(): raise asyncio.CancelledError
            await asyncio.sleep(0)
            print(f"writing to pulse audio: {sentance=}")
            audio_server.write(audio)    
            # if asyncio.current_task().cancelling(): raise asyncio.CancelledError
            await asyncio.sleep(0)
            print("writing pause")
            audio_server.write(sentance_pause)
            print("finished writing to pulse audio")
            
        audio_server.drain()
    except asyncio.CancelledError:
        print("cancelling play_audio()")
        #print("flushing pulse audio")
        #audio_server.flush()
    


async def read_from_fifo(loop):
    executor = ThreadPoolExecutor(max_workers=1)
    def open_read():
        with open('/tmp/kak_tts_fifo', "rb") as fifo:
            bytes = fifo.read()
            return bytes.decode('utf-8')

    while True:
        print("starting reading from fifo")
        data = await loop.run_in_executor(executor, open_read)
        print("finished reading from fifo")
        if not data: 
            print("not fifo data")
            continue
        print("starting decodemessage")
        message, remainder = DecodeMessage(data)
        print("ending decodemessage")
        assert remainder == ""
        yield message


async def main():
    print("main starting")
    loop = asyncio.get_running_loop()
    tokenizer = load(f"tokenizers/punkt/english.pickle")
    current_playing_buffer = None
    running_task = None

    async for message in read_from_fifo(loop):

        match message:
            case {"method": "write_to_file", "params":{"buffer": buffer, "bufname": bufname}}:
                print("starting write to file")
                spans = list(tokenizer.span_tokenize(buffer))
                sentances = tuple(buffer[s:e] for s, e in spans)
                audio = tuple(get_audio(s) for s in sentances)
                write_to_file(tuple(zip(audio, sentances)), sample_rate, pathlib.Path(bufname))

            case {"method": "narrate_from_cursor", "params":{"buffer": buffer, "cursor_byte_offset":cursor_byte_offset}}:
                # if buffer == current_playing_buffer: continue
                if running_task is not None: running_task.cancel()
                current_playing_buffer = buffer
                # to_cursor = buffer.encode('utf-8')[:cursor_byte_offset].decode('utf-8', errors='ignore')
                # cursor = len(to_cursor)
                cursor = cursor_byte_offset

                spans = list(tokenizer.span_tokenize(buffer))

                def find_span_index(cursor, spans):
                    for index, (start, end) in enumerate(spans):
                        if start <= cursor <= end:
                            return index
                    raise ValueError("cursor outside of buffer")
        
                start_sentance = find_span_index(cursor, spans)
                sentances = [buffer[s:e] for s, e in spans[start_sentance:]]
        
                print(f"{cursor=}\n{spans=}\n{sentances=}")

                running_task = asyncio.create_task(play_audio(sentances))

            case {"method": "cancel"}:
                if running_task is not None: 
                    print("cancelling task")
                    running_task.cancel()
                    audio_server.flush()

asyncio.run(main())