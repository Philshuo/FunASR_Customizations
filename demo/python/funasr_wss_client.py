# -*- encoding: utf-8 -*-
import os
import time
import websockets, ssl
import asyncio
# import threading
import argparse
import json
import traceback
from multiprocessing import Process
# from funasr.fileio.datadir_writer import DatadirWriter

import logging
import re
logging.basicConfig(filename='284log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    type=str,
                    default="localhost",
                    required=False,
                    help="host ip, localhost, 0.0.0.0")
parser.add_argument("--port",
                    type=int,
                    default=10095,
                    required=False,
                    help="grpc server port")
parser.add_argument("--chunk_size",
                    type=str,
                    default="5, 10, 5",
                    help="chunk")
parser.add_argument("--chunk_interval",
                    type=int,
                    default=10,
                    help="chunk")
parser.add_argument("--hotword",
                    type=str,
                    default="",
                    help="hotword file path, one hotword perline (e.g.:阿里巴巴 20)")
parser.add_argument("--audio_in",
                    type=str,
                    default=None,
                    help="audio_in")
parser.add_argument("--send_without_sleep",
                    action="store_true",
                    default=True,
                    help="if audio_in is set, send_without_sleep")
parser.add_argument("--thread_num",
                    type=int,
                    default=1,
                    help="thread_num")
parser.add_argument("--words_max_print",
                    type=int,
                    default=10000,
                    help="chunk")
parser.add_argument("--output_dir",
                    type=str,
                    default=None,
                    help="output_dir")
parser.add_argument("--ssl",
                    type=int,
                    default=1,
                    help="1 for ssl connect, 0 for no ssl")
parser.add_argument("--use_itn",
                    type=int,
                    default=1,
                    help="1 for using itn, 0 for not itn")
parser.add_argument("--mode",
                    type=str,
                    default="2pass",
                    help="offline, online, 2pass")

args = parser.parse_args()
args.chunk_size = [int(x) for x in args.chunk_size.split(",")]
print(args)
# voices = asyncio.Queue()
from queue import Queue

voices = Queue()
offline_msg_done=False

if args.output_dir is not None:
    # if os.path.exists(args.output_dir):
    #     os.remove(args.output_dir)
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

# mapping: from Chinese num to Arabic num
mapping = {
    "洞": "0", "动": "0", "栋": "0", "幺": "1", "腰": "1", "两": "2", "梁": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "拐": "7", "馆": "7", "八": "8", "勾": "9", "沟": "9", "九": "9"
}

def insert_punc(text, online_text):
    # find positions of punctuation in text
    punctuation_positions = [m.start() for m in re.finditer(r'[^\w\s]', text)]

    temp = online_text
    # insert punctuatiions of text into temp
    for i, char in enumerate(text):
        if i in punctuation_positions:
            temp = temp[:i] + char + temp[i:]
    return temp

def replace_chinese_numbers(text):
    # define Regular Expression
    pattern = re.compile(r'[一二两三四五六七八九零百十千万0-9]+')

    # find and replace the Chinese number with arabic
    result = pattern.sub(lambda x: chinese_to_arabic(x.group()), text)
    return result

def chinese_to_arabic(s):
    chinese_nums = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    unit_map = {'十': 10, '百': 100, '千': 1000, '万': 10000}
    result = 0; temp_result = 0; unit = 1; index = 0; length = len(s)

    while index < length:
        char = s[index]
        if char not in chinese_nums and char not in unit_map and not(char.isdigit()):
            index += 1
        else:
            if char == '十':
                if (index+1) < length:
                    if s[index+1] in chinese_nums:
                        result = 10 + chinese_nums[s[index+1]]
                    else:
                        result = 10 + int(s[index+1])
                else:
                    result = 10
                break
            else:
                if not char.isdigit():
                    temp_result = chinese_nums[char]
                else:
                    temp_result = int(char)

                if (index + 1 < length) and (s[index + 1] in unit_map):
                    unit = unit_map[s[index + 1]]
                    result += temp_result * unit
                    index += 2
                else:
                    unit = max(int(unit/10), 1)
                    result += temp_result * unit
                    index += 1
                    if index+1 == length and unit != 1: # 
                        print(str(result)+s[index])
                        return str(result)+s[index]

    return str(result)
  
# post process func
def post_process_online(text, online_text):
    # time-minutes
    matches_time = re.finditer(r'(时)(.*?)(分)', text)
    offset = 0
    for match in matches_time:
        temp = replace_chinese_numbers(match.group(2))
        text = text[:match.start(2) + offset] + temp + text[match.end(2) + offset:]
        online_text = online_text[:match.start(2) + offset] + temp + online_text[match.end(2) + offset:]
        offset += len(temp) - len(match.group(2))
    #print(text)
    
    # time-hours
    matches_hour = re.finditer(r'.{3}时', text)
    if matches_hour:
        # 从“零”到“二十四”
        for match in matches_hour:
            temp = text[match.start():match.end()]
            temp = replace_chinese_numbers(temp)
            text = text[:match.start()]+temp+text[match.end():]
            online_text = online_text[:match.start()]+temp+online_text[match.end():]
    #print(text)

    # degree
    matches_degree = re.finditer(r'.{3}度', text)#高度和角度
    if matches_degree:
        for match in matches_degree:
            temp = match.group()
            match_gao = re.search(r'高(.*)', temp)
            if match_gao:
                break
            else:# print(temp)
                if temp[-2] != "速": # 度前面没有“速”
                    temp = replace_chinese_numbers(temp)
                    # print(temp)
                    text = text.replace(match.group(), temp)
                    online_text = online_text[:match.start()]+temp+online_text[match.end():]
    # print(online_text)
    
    # distance
    matches_dis = list(re.finditer(r'.{2,5}公里', text))
    new_text_parts = []
    online_text_parts = []
    last_end = 0
    if matches_dis:
        for match in matches_dis:
        #print("公里")
            chinese_numbers = match.group()[:-2]  # 提取“公里”前的文本
            match_du = re.search(r'度(.*)', chinese_numbers)
            if match_du:
                res_str = chinese_numbers[:match_du.start(1)]
                chinese_numbers = match_du.group(1)
                arabic_numbers = replace_chinese_numbers(chinese_numbers)
                arabic_numbers = res_str+arabic_numbers
            else:
            #print(chinese_numbers)
                arabic_numbers = replace_chinese_numbers(chinese_numbers)
        #print(arabic_numbers)
            new_text_parts.append(text[last_end:match.start()] + arabic_numbers + "公里")
            online_text_parts.append(online_text[last_end:match.start()] + arabic_numbers + "公里")
        #print(new_text_parts)
            last_end = match.end()
        new_text_parts.append(text[matches_dis[-1].end():])
        online_text_parts.append(online_text[matches_dis[-1].end():])
    else:
        new_text_parts.append(text)
        online_text_parts.append(online_text)
    new_text = ''.join(new_text_parts)
    new_online_text = ''.join(online_text_parts)

    # velocity
    matches_vel = re.finditer(r'速度(.{2,5})', new_text)
    online_parts = []
    last_end_vel = 0  # 重新计算速度部分的最后结束位置
    for match in matches_vel:
        temp = replace_chinese_numbers(match.group())
        online_parts.append(new_online_text[last_end_vel:match.start()] + temp)
        last_end_vel = match.end() + 1 # 往后多移一位，不然会重复
    online_vel_text = ''.join(online_parts) + new_online_text[last_end_vel:]
    # print(online_vel_text)
    
    # print(online_text)
    online_text = online_vel_text if online_vel_text else online_text
    for key, value in mapping.items():
        online_text = online_text.replace(key, value)
    
    return str(text), str(online_text)


async def record_microphone():
    is_finished = False
    import pyaudio
    # print("2")
    global voices
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    chunk_size = 60 * args.chunk_size[1] / args.chunk_interval
    CHUNK = int(RATE / 1000 * chunk_size)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    # hotwords
    fst_dict = {}
    hotword_msg = ""
    if args.hotword.strip() != "":
        f_scp = open(args.hotword)
        hot_lines = f_scp.readlines()
        for line in hot_lines:
            words = line.strip().split(" ")
            if len(words) < 2:
                print("Please checkout format of hotwords")
                continue
            try:
                fst_dict[" ".join(words[:-1])] = int(words[-1])
            except ValueError:
                print("Please checkout format of hotwords")
        hotword_msg=json.dumps(fst_dict)

    use_itn=True
    if args.use_itn == 0:
        use_itn=False
    
    message = json.dumps({"mode": args.mode, "chunk_size": args.chunk_size, "chunk_interval": args.chunk_interval,
                          "wav_name": "microphone", "is_speaking": True, "hotwords":hotword_msg, "itn": use_itn})
    #voices.put(message)
    await websocket.send(message)
    while True:
        data = stream.read(CHUNK)
        message = data
        #voices.put(message)
        await websocket.send(message)
        await asyncio.sleep(0.005)

async def record_from_scp(chunk_begin, chunk_size):
    global voices
    is_finished = False
    if args.audio_in.endswith(".scp"):
        f_scp = open(args.audio_in)
        wavs = f_scp.readlines()
    else:
        wavs = [args.audio_in]

    # hotwords
    fst_dict = {}
    hotword_msg = ""
    if args.hotword.strip() != "":
        f_scp = open(args.hotword)
        hot_lines = f_scp.readlines()
        for line in hot_lines:
            words = line.strip().split(" ")
            if len(words) < 2:
                print("Please checkout format of hotwords")
                continue
            try:
                fst_dict[" ".join(words[:-1])] = int(words[-1])
            except ValueError:
                print("Please checkout format of hotwords")
        hotword_msg=json.dumps(fst_dict)
        print (hotword_msg)

    sample_rate = 16000
    wav_format = "pcm"
    use_itn=True
    if args.use_itn == 0:
        use_itn=False
     
    if chunk_size > 0:
        wavs = wavs[chunk_begin:chunk_begin + chunk_size]
    for wav in wavs:
        wav_splits = wav.strip().split()
 
        wav_name = wav_splits[0] if len(wav_splits) > 1 else "demo"
        wav_path = wav_splits[1] if len(wav_splits) > 1 else wav_splits[0]
        if not len(wav_path.strip())>0:
           continue
        if wav_path.endswith(".pcm"):
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()
        elif wav_path.endswith(".wav"):
            import wave
            with wave.open(wav_path, "rb") as wav_file:
                params = wav_file.getparams()
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                audio_bytes = bytes(frames)
        else:
            wav_format = "others"
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()

        # stride = int(args.chunk_size/1000*16000*2)
        stride = int(60 * args.chunk_size[1] / args.chunk_interval / 1000 * 16000 * 2)
        chunk_num = (len(audio_bytes) - 1) // stride + 1
        # print(stride)

        # send first time
        message = json.dumps({"mode": args.mode, "chunk_size": args.chunk_size, "chunk_interval": args.chunk_interval, "audio_fs":sample_rate,
                          "wav_name": wav_name, "wav_format": wav_format, "is_speaking": True, "hotwords":hotword_msg, "itn": use_itn})

        #voices.put(message)
        await websocket.send(message)
        is_speaking = True
        for i in range(chunk_num):

            beg = i * stride
            data = audio_bytes[beg:beg + stride]
            message = data
            #voices.put(message)
            await websocket.send(message)
            if i == chunk_num - 1:
                is_speaking = False
                message = json.dumps({"is_speaking": is_speaking})
                #voices.put(message)
                await websocket.send(message)
 
            sleep_duration = 0.001 if args.mode == "offline" else 60 * args.chunk_size[1] / args.chunk_interval / 1000
            
            await asyncio.sleep(sleep_duration)
    
    if not args.mode=="offline":
        await asyncio.sleep(2)
    # offline model need to wait for message recved
    
    if args.mode=="offline":
      global offline_msg_done
      while  not  offline_msg_done:
         await asyncio.sleep(1)
    
    await websocket.close()


          
async def message(id):
    global websocket,voices,offline_msg_done
    text_print = ""
    text_print_2pass_online = ""
    text_print_2pass_offline = ""
    text_online = ""
    text_print_final = ""
    logger = logging.getLogger(__name__)
    if args.output_dir is not None:
        ibest_writer = open(os.path.join(args.output_dir, "text.{}".format(id)), "a", encoding="utf-8")
    else:
        ibest_writer = None
    try:
       while True:
        
            meg = await websocket.recv()
            meg = json.loads(meg)
            wav_name = meg.get("wav_name", "demo")
            text = meg["text"]
            timestamp=""
            if "timestamp" in meg:
                timestamp = meg["timestamp"]

            if ibest_writer is not None:
                if timestamp !="":
                    text_write_line = "{}\t{}\t{}\n".format(wav_name, text, timestamp)
                else:
                    text_write_line = "{}\t{}\n".format(wav_name, text)
                ibest_writer.write(text_write_line)
                
            if meg["mode"] == "online":
                text_print += "{}".format(text)
                text_print = text_print[-args.words_max_print:]
                os.system('clear')
                print("\rpid" + str(id) + ": " + text_print)
            elif meg["mode"] == "offline":
                if timestamp !="":
                    text_print += "{} timestamp: {}".format(text, timestamp)
                else:
                    text_print += "{}".format(text)

                # text_print = text_print[-args.words_max_print:]
                # os.system('clear')
                print("\rpid" + str(id) + ": " + wav_name + ": " + text_print)
                offline_msg_done = True
            else:
                if meg["mode"] == "2pass-online":
                    text_print_2pass_online += "{}".format(text)
                    text_print = text_print_2pass_offline + text_print_2pass_online
                    text_online += "{}".format(text)
                    # os.system('clear')
                    # print("\rONline-format_text" + ": " + "{}".format(text))
                    # print("\rONline-text_online" + ": " + text_online)
                    # print("\rONline-text_print_2pass_online" + ": " + text_print_2pass_online)
                    # print("\rONline-text_print_2pass_offline" + ": " + text_print_2pass_offline)
                    # print("\rONline-text_print" + ": " + text_print)
                else:
                    text_print_2pass_online = ""
                    # text_print = text_print_2pass_offline + "{}".format(text)
                    temp_res = "{}".format(text)
                    logger.info(f'format text: {temp_res}, text_online: {text_online}')
                    result = insert_punc(temp_res, text_online)
                    logger.info(f"insert_punc text: {result}")
                    result = result + temp_res[len(result):]
                    logger.info(f"alignment text: {result}")
                    temp_res, result = post_process_online(temp_res, result)# 拿到了结果，要累加起来了，即，送给2pass_offline
                    logger.info(f'after post_process_online: text = {temp_res}, result = {result}')
                    # os.system('clear')
                    # print("\rOFFline-format_text" + ": " + "{}".format(text))
                    # print("\rOFFline-result" + ": " + result)
                    # print("\rOFFline-text_online" + ": " + text_online)
                    # print("\rtext length" + ": " + str(len("{}".format(text))))
                    # print("\ronline length" + ": " + str(len(text_online)))
                    # print("\rOFFline-text_print" + ": " + text_print)
                    # text_print_2pass_offline += "{}".format(text) + " " #添加空格进行隔断，避免数字+编号，混在一起
                    text_print_2pass_offline += result + " "
                    text_print = text_print_2pass_offline.replace(" ", "")
                    # print("\rOFFline-2pass_offline" + ": " + text_print_2pass_offline)
                    # print("\rOFFline-2pass_online" + ": " + text_print_2pass_online)
                    # text_post_process = post_process(text_print) #可以放这儿
                    # text_print_final = text_post_process.replace(" ", "")
                    text_online = "" # 每次offline之后清空online的数据，以便和下一组的offline对比
                text_print = text_print[-args.words_max_print:]
                #text_print = text_print_2pass_offline[-args.words_max_print:]
                # text_post_process = post_process(text_print) #可以放这儿
                #text_print_final = text_post_process.replace(" ", "")
                os.system('clear')
                print("\rpid" + str(id) + ": " + text_print)
                # print("\rfinal" + str(id) + ": " + text_print_final)
                offline_msg_done=True
                
                

    except Exception as e:
            print("Exception:", e)
            traceback.print_exc() # 打开详细的追踪信息，异常发生时的堆栈跟踪
            await websocket.close() # 发生异常时关闭websocket连接




async def ws_client(id, chunk_begin, chunk_size):
  if args.audio_in is None:
       chunk_begin=0
       chunk_size=1
  global websocket,voices,offline_msg_done
 
  for i in range(chunk_begin,chunk_begin+chunk_size):
    offline_msg_done=False
    voices = Queue()
    if args.ssl == 1:
        ssl_context = ssl.SSLContext()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        uri = "wss://{}:{}".format(args.host, args.port)
    else:
        uri = "ws://{}:{}".format(args.host, args.port)
        ssl_context = None
    print("connect to", uri)
    async with websockets.connect(uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context) as websocket:
        if args.audio_in is not None:
            task = asyncio.create_task(record_from_scp(i, 1))
        else:
            task = asyncio.create_task(record_microphone())
        task3 = asyncio.create_task(message(str(id)+"_"+str(i))) #processid+fileid
        await asyncio.gather(task, task3)
  exit(0)
    

def one_thread(id, chunk_begin, chunk_size):
    asyncio.get_event_loop().run_until_complete(ws_client(id, chunk_begin, chunk_size))
    asyncio.get_event_loop().run_forever()

if __name__ == '__main__':
    # for microphone
    if args.audio_in is None:
        p = Process(target=one_thread, args=(0, 0, 0))
        p.start()
        p.join()
        print('end')
    else:
        # calculate the number of wavs for each preocess
        if args.audio_in.endswith(".scp"):
            f_scp = open(args.audio_in)
            wavs = f_scp.readlines()
        else:
            wavs = [args.audio_in]
        for wav in wavs:
            wav_splits = wav.strip().split()
            wav_name = wav_splits[0] if len(wav_splits) > 1 else "demo"
            wav_path = wav_splits[1] if len(wav_splits) > 1 else wav_splits[0]
            audio_type = os.path.splitext(wav_path)[-1].lower()


        total_len = len(wavs)
        if total_len >= args.thread_num:
            chunk_size = int(total_len / args.thread_num)
            remain_wavs = total_len - chunk_size * args.thread_num
        else:
            chunk_size = 1
            remain_wavs = 0

        process_list = []
        chunk_begin = 0
        for i in range(args.thread_num):
            now_chunk_size = chunk_size
            if remain_wavs > 0:
                now_chunk_size = chunk_size + 1
                remain_wavs = remain_wavs - 1
            # process i handle wavs at chunk_begin and size of now_chunk_size
            p = Process(target=one_thread, args=(i, chunk_begin, now_chunk_size))
            chunk_begin = chunk_begin + now_chunk_size
            p.start()
            process_list.append(p)

        for i in process_list:
            p.join()

        print('end')
