class configuration:
    def general_config(self,required):
        gc={"model_name":"google/siglip2-base-patch16-224","dbname":"vision_rag","device":"mps",
            "supported_format":{"Audio":[".mp3"],"video":[".mp4"],"photo":[".jpeg",".png",".jpg"],"doc":[".pdf",".docx"]},
            "llm_model":"Qwen/Qwen3-VL-8B-Instruct"}
        return gc[required]
