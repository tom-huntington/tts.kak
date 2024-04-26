import os
from contextlib import contextmanager
import pasimple
from TTS.tts.utils.synthesis import synthesis
from nltk.data import load
import json
import asyncio

fifo_path = '/tmp/kak_tts_fifo'
channels = 1  # Mono audio

cache = {}
from TTS.api import TTS
tts = TTS(model_name="tts_models/en/vctk/vits", gpu=True)
sample_rate = tts.synthesizer.output_sample_rate

format = pasimple.PA_SAMPLE_FLOAT32LE  # Little-endian 16-bit
audio_server = pasimple.PaSimple(pasimple.PA_STREAM_PLAYBACK, format, channels, sample_rate)

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



def get_audio(sentance):
    print("get_audio")
    if not sentance in cache:
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
        loop = asyncio.get_running_loop()
        for sentance in sentances:
            print("starting get audio")
            # if asyncio.current_task().cancelling(): raise asyncio.CancelledError
            await asyncio.sleep(0)
            audio = await asyncio.shield(get_audio_async(sentance))
            print("finishing get audio")
            # if asyncio.current_task().cancelling(): raise asyncio.CancelledError
            print(f"writing to pulse audio: {sentance=}")
            # await loop.run_in_executor(None, audio_server.write, audio)    
            # audio_server.write(audio)
            audio += sentance_pause
            step=22050//4
            stop = len(audio)

            for i in range(0, stop, step):
                await asyncio.sleep(0)
                await loop.run_in_executor(None, audio_server.write, audio[i:i+step]) 
                # audio_server.write(audio[i:i+step])
            # if asyncio.current_task().cancelling(): raise asyncio.CancelledError
            # await asyncio.sleep(0)
            # print("writing pause")
            # audio_server.write(sentance_pause)
            # print("finished writing to pulse audio")
            
        audio_server.drain()
    except asyncio.CancelledError:
        print("cancelling play_audio()")
        #print("flushing pulse audio")
        #audio_server.flush()
    


async def read_from_fifo(loop):
    # executor = ThreadPoolExecutor(max_workers=1)
    def open_read():
        with open(fifo_path, "rb") as fifo:
            print("start fifo.read")
            bytes = fifo.read()
            print("fin fifo.read -- start bytes.decode")
            s = bytes.decode('utf-8')
            print("fin bytes.decode")
            return s

    while True:
        # print("starting reading from fifo")
        data = await loop.run_in_executor(None, open_read)
        print("finished reading from fifo")
        if not data: 
            # print("not fifo data")
            continue
        # print("starting decodemessage")
        message, remainder = DecodeMessage(data)
        # # # print("ending decodemessage")
        assert remainder == ""
        yield message


async def main2():
    print("main starting")
    loop = asyncio.get_running_loop()
    tokenizer = load(f"tokenizers/punkt/english.pickle")
    current_playing_buffer = None
    running_task = None

    async for message in read_from_fifo(loop):
        print("matching new message ")
        match message:
            case {"method": "narrate_from_cursor", "params":{"buffer": buffer, "cursor_byte_offset":cursor_byte_offset}}:
                # if buffer == current_playing_buffer: continue
                if running_task is not None: running_task.cancel()
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


def main():
    if not os.path.exists(fifo_path):
        os.mkfifo(fifo_path)
    asyncio.run(main2())

if __name__ == "__main__":
    main()