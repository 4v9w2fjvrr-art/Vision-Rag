
import MNP_all
from pathlib import Path

a=MNP_all.retriveing()
a.run_once()
#a.img_processing(a.load_document([str(i) for i in Path("/Users/sriramapadmanabhan/vision-rag/data/Valencia Tour/Photos").iterdir()],a.supported_formats))

#a.video_processing(a.load_document([str(i) for i in Path("/Users/sriramapadmanabhan/vision-rag/data/Valencia Tour/Videos").iterdir()],a.supported_formats))
a.extract_audio(a.load_document([str(i) for i in Path("/Users/sriramapadmanabhan/vision-rag/data/Valencia Tour/Videos").iterdir()],a.supported_formats))
#user_query=input("Enter your search query: ")
#b=a.user_query_embeding(user_query)
#c=a.retrieve_embedding(b)
#a.evaluvation(c)
#lc=llm_call()
#lc.use_llm(c)
