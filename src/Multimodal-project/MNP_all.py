from pathlib import Path
import re
from pydantic import validate_call
import psycopg,cv2,pytesseract,subprocess
import torch,MNP_settings
from PIL import Image
from pgvector.psycopg import register_vector
from transformers import AutoModel, AutoProcessor,AutoModelForImageTextToText
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

class prerequeist:
    def run_once(self):
        self.MODEL_NAME = MNP_settings.configuration().general_config("model_name")
        self.db_name = MNP_settings.configuration().general_config("dbname")
        self.device = MNP_settings.configuration().general_config("device")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device).eval()
        self.whisper_model_name = "small"
        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.whisper_model = WhisperModel(self.whisper_model_name,device=self.device,compute_type="int8")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.conn = psycopg.connect(dbname=self.db_name)
        self.supported_formats=sum(MNP_settings.configuration().general_config("supported_format").values(),[])
        register_vector(self.conn)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS images (id BIGSERIAL PRIMARY KEY,path TEXT UNIQUE NOT NULL,filename TEXT NOT NULL,model TEXT NOT NULL,embedding VECTOR(768) NOT NULL,ocr_text TEXT NULL,ocr_embedding VECTOR(768) NULL)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS video_frames (id BIGSERIAL PRIMARY KEY,video_path TEXT NOT NULL,filename TEXT NOT NULL,frame_number BIGINT NOT NULL,timestamp DOUBLE PRECISION NOT NULL,model TEXT NOT NULL,embedding VECTOR(768) NOT NULL,UNIQUE(video_path, frame_number),ocr_text TEXT NULL,ocr_embedding VECTOR(768) NULL)""")
        #self.conn.execute("""CREATE TABLE IF NOT EXISTS text (id BIGSERIAL PRIMARY KEY,path TEXT UNIQUE NOT NULL,filename TEXT NOT NULL,model TEXT NOT NULL,embedding VECTOR(768) NOT NULL,text TEXT NULL,text_embedding VECTOR(768) NULL)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS audio_segments (id BIGSERIAL PRIMARY KEY,source_path TEXT NOT NULL,filename TEXT NOT NULL,start_time DOUBLE PRECISION NOT NULL,end_time DOUBLE PRECISION NOT NULL,transcript TEXT,model TEXT NOT NULL,embedding VECTOR(768),UNIQUE(source_path, start_time, end_time))""")
        self.conn.commit()


    @validate_call
    def load_document(self,path:list[str],supported_formats:list[str]):
        return_values=[]
        for i in path:
            file = Path(i)
            if file.is_file() and file.suffix.lower() in supported_formats:
                return_values.append({"path":i,"name":file.name,"format":file.suffix.lower()})
            else:
                print("Not valid document")
        return return_values

class img_process(prerequeist):
    def img_processing(self,img_list):
        for i in img_list:
            if i["format"] in MNP_settings.configuration().general_config("supported_format")["photo"]:
                self.image = Image.open(Path(i["path"])).convert("RGB")
                self.inputs = self.processor(images=self.image, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    self.output = self.model.get_image_features(**self.inputs)
                    self.embedding = self.output.pooler_output
                #embedding = self.embedding[0].float().cpu().numpy()
                embedding = self.embedding.detach().float().cpu().numpy().reshape(-1)
                embedding /= (embedding ** 2).sum() ** 0.5

                text = pytesseract.image_to_string(self.image).strip()
                inputs_ocr = self.processor(text=[text],return_tensors="pt",padding="max_length",truncation=True,max_length=64).to(self.device)
                with torch.inference_mode():
                    output_ocr = self.model.get_text_features(**inputs_ocr)
                    embedding_ocr = output_ocr.pooler_output
                embedding_ocr = (embedding_ocr.detach().float().cpu().numpy().reshape(-1))
                embedding_ocr /= (embedding_ocr ** 2).sum() ** 0.5
                self.conn.execute("""INSERT INTO images (path, filename, model, embedding, ocr_text, ocr_embedding) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (path) DO UPDATE SET filename = EXCLUDED.filename,model = EXCLUDED.model,embedding = EXCLUDED.embedding,ocr_text = EXCLUDED.ocr_text,ocr_embedding = EXCLUDED.ocr_embedding""", (str(Path(i["path"]).resolve()), Path(i["path"]).name, self.MODEL_NAME, embedding,text,embedding_ocr))
                print(f"{i['name']}-> Image Embedding done")
                print(f"{i['name']}-> OCR Processing done")
        self.conn.commit()
        print("Image embedded and saved successfully")

class video_process(img_process):
    def video_processing(self, video_list, interval_seconds=2):
        for i in video_list:
            if i["format"] in MNP_settings.configuration().general_config("supported_format")["video"]:
                video_path = i["path"]
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = int(fps * interval_seconds)
                frame_number = 0
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    if frame_number % frame_interval == 0:
                        timestamp = frame_number / fps
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame)
                        inputs = self.processor(images=image,return_tensors="pt").to(self.device)
                        with torch.inference_mode():
                            output = self.model.get_image_features(**inputs)
                            embedding = output.pooler_output
                        embedding = (embedding.detach().float().cpu().numpy().reshape(-1))
                        embedding /= (embedding ** 2).sum() ** 0.5

                        text = pytesseract.image_to_string(image).strip()
                        inputs_ocr = self.processor(text=[text],return_tensors="pt",padding="max_length",truncation=True,max_length=64).to(self.device)
                        with torch.inference_mode():
                            output_ocr = self.model.get_text_features(**inputs_ocr)
                            embedding_ocr = output_ocr.pooler_output
                        embedding_ocr = (embedding_ocr.detach().float().cpu().numpy().reshape(-1))
                        embedding_ocr /= (embedding_ocr ** 2).sum() ** 0.5

                        self.conn.execute("""INSERT INTO video_frames (video_path,filename,frame_number,timestamp,model,embedding, ocr_text,ocr_embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (video_path,frame_number) DO UPDATE SET filename = EXCLUDED.filename,model = EXCLUDED.model,embedding = EXCLUDED.embedding,ocr_text = EXCLUDED.ocr_text,ocr_embedding = EXCLUDED.ocr_embedding""", (str(Path(video_path).resolve()),Path(video_path).name,frame_number,timestamp,self.MODEL_NAME,embedding,text,embedding_ocr))
                        print(f"{i['name']}-> Video Embedding done")
                        print(f"{i['name']}-> OCR Embedding done")
                    frame_number += 1
                cap.release()
        self.conn.commit()
        self.conn.close()
        print("Video frames embedded and saved successfully")

#class pdf_process(prerequeist):
class audio_process(prerequeist):
    def extract_audio(self, video_list):
        for i in video_list:
            self.video_path = Path(i)
        self.audio_path = self.video_path.with_suffix(".wav")
        print(self.audio_path)
        subprocess.run(["ffmpeg","-y","-i", str(self.video_path),"-vn","-acodec", "pcm_s16le",str(self.audio_path)],check=True)
        return str(self.audio_path)


        def transcribe_audio(self, audio_path):
            segments, info = self.whisper_model.transcribe(audio_path,beam_size=5)
            results = []
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                results.append({"start_time": float(segment.start),"end_time": float(segment.end),"text": text})

            return results


        def embed_and_store(self,source_path,segments):
            source_path = Path(source_path)
            for segment in segments:
                embedding = self.embedding_model.encode(segment["text"],normalize_embeddings=True)
                self.conn.execute("""INSERT INTO audio_segments(source_path,filename,start_time,end_time,transcript,whisper_model,embedding_model,embedding)
                                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (source_path,start_time,end_time)
                                            DO UPDATE SET
                                            transcript = EXCLUDED.transcript,whisper_model =EXCLUDED.whisper_model,embedding_model =EXCLUDED.embedding_model,embedding =EXCLUDED.embedding""",
                    (str(source_path.resolve()),source_path.name,segment["start_time"],segment["end_time"],segment["text"],self.whisper_model_name,self.embedding_model_name,embedding))
            self.conn.commit()


        def process_video(self, video_path):
            print(f"Processing video: {video_path}")
            audio_path = self.extract_audio(video_path)
            print(f"Audio extracted: {audio_path}")
            segments = self.transcribe_audio(audio_path)
            for segment in segments:
                print(segment["start_time"],segment["end_time"],segment["text"])
            self.embed_and_store(video_path,segments)
            print("Audio transcript embedded and stored")


        def process_audio(self, audio_path):

            print(f"Processing audio: {audio_path}")

            segments = self.transcribe_audio(audio_path)

            self.embed_and_store(audio_path,segments)

            print(
                "Audio transcript embedded and stored"
            )


        def query_embedding(self, query):

            embedding = self.embedding_model.encode(
                query,
                normalize_embeddings=True
            )

            return embedding


        def retrieve_audio(
            self,
            query,
            limit=5
        ):

            query_embedding = self.query_embedding(
                query
            )

            results = self.conn.execute(
                """
                SELECT

                    id,
                    source_path,
                    filename,

                    start_time,
                    end_time,

                    transcript,

                    1 - (embedding <=> %s)
                        AS similarity

                FROM audio_segments

                ORDER BY
                    embedding <=> %s

                LIMIT %s
                """,

                (
                    query_embedding,
                    query_embedding,
                    limit
                )
            ).fetchall()

            final_results = []

            for row in results:

                final_results.append(
                    {
                        "id": row[0],

                        "path": row[1],
                        "filename": row[2],

                        "start_time": row[3],
                        "end_time": row[4],

                        "transcript": row[5],

                        "similarity": float(row[6])
                    }
                )

            return final_results


        def close(self):

            if self.conn:
                self.conn.close()


    # --------------------------------------------------
    # RUN
    # --------------------------------------------------

    #a = audio_process()

    #a.run_once()


    # --------------------------------------------------
    # INGEST VIDEO
    # --------------------------------------------------

    video_path = (
        "/Users/sriramapadmanabhan/"
        "vision-rag/data/Valencia Tour/Videos/"
        "MP4_20240930_012119VLOG.mp4"
    )

    # Run once when indexing
    # a.process_video(video_path)


    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    query = input(
        "Enter audio search query: "
    )

    results = a.retrieve_audio(
        query,
        limit=5
    )

    for result in results:

        print(
            "\n"
            "File       :", result["filename"],
            "\nStart Time :", result["start_time"],
            "\nEnd Time   :", result["end_time"],
            "\nTranscript :", result["transcript"],
            "\nSimilarity :", result["similarity"]
        )


    a.close()


class retriveing(video_process,audio_process):
    def user_query_embeding(self,query):
        if not self.conn:
            self.conn = psycopg.connect(dbname=self.db_name)
            register_vector(self.conn)
        inputs = self.processor(text=[query],return_tensors="pt",padding="max_length",truncation=True,max_length=64).to(self.device)
        with torch.inference_mode():
            output = self.model.get_text_features(**inputs)
            embedding = output.pooler_output
            embedding = embedding.detach().float().cpu().numpy().reshape(-1)
            embedding /= (embedding ** 2).sum() ** 0.5
            return {"user_query":query,"embedding":embedding}

    def retrieve_embedding(self,query_embedings,limit=7,threshold=0.10):
        query_embeding=query_embedings["embedding"]
        user_query=query_embedings['user_query']
        final_list=[]
        img_result = self.conn.execute(""" SELECT id,path,filename,1 - (embedding <=> %s) AS similarity FROM images
                                            WHERE 1 - (embedding <=> %s) >= %s
                                            ORDER BY embedding <=> %s LIMIT %s""",
                                                (query_embeding, query_embeding,threshold,query_embeding, limit)).fetchall()
        #for i in img_result:
        #    print(i[-1])
        video_result = self.conn.execute(""" SELECT id,video_path,filename,frame_number,timestamp,1 - (embedding <=> %s) AS similarity FROM video_frames
                                                WHERE 1 - (embedding <=> %s) >= %s
                                                ORDER BY embedding <=> %s LIMIT %s
                                                """,
                                                (query_embeding, query_embeding,threshold,query_embeding, limit)).fetchall()
        #print(video_result)

        ocr_result = self.conn.execute("""SELECT id,path,filename,ocr_text,1 - (ocr_embedding <=> %s) AS similarity FROM images
                                                WHERE 1 - (embedding <=> %s) >= %s and  ocr_embedding IS NOT NULL
                                                ORDER BY ocr_embedding <=> %s LIMIT %s""",
                                                (query_embeding, query_embeding,threshold,query_embeding, limit)).fetchall()
        #print(ocr_result)

        for k,i in {"image":img_result,"Video":video_result,"img_ocr":ocr_result}.items():
            for j in i:
                if k =="image":
                    final_list.append({"user_query":user_query,"type": "Image","path": j[1],"filename": j[2],"frame_number": None,"timestamp": None,"similarity": j[-1]})
                elif k=="Video":
                    final_list.append({"user_query":user_query,"type": "Video","path": j[1],"filename": j[2],"frame_number": j[3],"timestamp": j[4],"similarity": j[-1]})
                elif k=="ocr_result":
                    final_list.append({"user_query":user_query,"type": "OCR_Image","path": j[1],"filename": j[2],"frame_number": j[3],"timestamp": j[4],"similarity": j[-1]})
        final_list.sort(key=lambda x:x["similarity"],reverse=True)
        for i in final_list[:10]:
            if i['type']=="Image":
                print(f"{i['type']} -> {i['path']}")
                subprocess.run(["open", i['path']])
            elif i['type']=="Video":
                print(f"haiiiiii {i['type']} -> {i['path']} -> {i['timestamp']}")
                #subprocess.run(["/Applications/VLC.app/Contents/MacOS/VLC","--start-time",str(i['timestamp']),i['path']])
        self.conn.close()

        return final_list

    def evaluvation(self,final_list):
        """
        | Metric | What are we checking? |
        |---|---|
        | Hit@K | Did we find **anything correct**? |
        | Recall@K | Did we find **all/most correct items**? |
        | Precision@K | Are the results we returned **mostly correct**? |
        | MRR | How quickly does the **first correct result** appear? |
        | nDCG@K | Are the **best matches ranked highest**? |
        | MAP | How good is ranking of **all relevant results across many queries**? |
        """

        retrived_count=0
        correct_count=0
        ground_truth={"group selfy using mirror":["1727639923219.jpg","1727639923229.jpg","1727639923240.jpg"],"food":["1727639923253.jpg"],"show menu":["1727639923265.jpg"]}
        for i in final_list:
            if  i['type']== "Image":
                retrived_count+=1
                if i['user_query'] not in ground_truth:
                    print("no ground truth")
                else:
                    if i['filename'] in ground_truth[i['user_query']]:
                        i['hit@k']=1
                        correct_count+=1
                    else:
                        i['hit@k']=0

class llm_call:
    def use_llm(self,final_result):
        self.content=[]
        self.llm_model = MNP_settings.configuration().general_config("llm_model")
        self.processor = AutoProcessor.from_pretrained(self.llm_model)
        self.model = AutoModelForImageTextToText.from_pretrained(self.llm_model,device_map="auto")
        for result in final_result:
            if result["type"] == "Image":
                image = Image.open(result["path"]).convert("RGB")
            elif result["type"] == "Video":
                cap = cv2.VideoCapture(result["path"])
                cap.set(cv2.CAP_PROP_POS_MSEC,result["timestamp"] * 1000)
                success, frame = cap.read()
                cap.release()
                if not success:
                    continue
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                continue

            self.content.append({"type": "image","image": image})

        self.content.append({
            "type": "text",
            "text": (
                f"User query: {user_query}\n"
                "These are candidate results. "
                "Return only the number of the best matching image/frame."
            )
        })


        """for result in final_result:
            image = Image.open(result["path"]).convert("RGB")
            self.content.append({"type": "image","image": image})
            self.content.append({"type": "text",
                "text": (f"User query: {user_query}\n" "These images are candidate results from retrieval. " "Return only the number of the image that best matches the query. "
                    "Image numbering starts from 1.")})"""

        messages = [{"role": "user","content": self.content}]
        inputs = self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs,max_new_tokens=20)
        generated_tokens = output[0][inputs["input_ids"].shape[1]:]
        answer = self.processor.decode(generated_tokens,skip_special_tokens=True).strip()
        best_index = int(answer) - 1
        best_result = final_result[best_index]
        print(best_result)
        return best_result
