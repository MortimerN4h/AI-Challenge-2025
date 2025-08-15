from PIL import Image
import faiss
import matplotlib.pyplot as plt
import math
import numpy as np 
import clip
from langdetect import detect

class Myfaiss: # chu yeu xu li load va tim kiem hinh anh 
    def __init__(self, bin_file : str,id2img_fps, device, translater, clip_backbone="ViT-B/32"):
        self.index= self.load_bin_file(bin_file)
        self.id2img_fps= id2img_fps
        self.device= device
        self.model, _ = clip.load(clip_backbone, device=device)
        self.translater = translater

    def load_bin_file(self, bin_file: str): # load anh index tu file .bin 
        return faiss.read_index(bin_file)
    
    def show_images(self, image_paths):
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))

        for i in range(1, columns*rows +1):
          img = plt.imread(image_paths[i - 1])
          ax = fig.add_subplot(rows, columns, i)
          ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))

          plt.imshow(img)
          plt.axis("off")

        plt.show()
        
    def image_search(self, id_query, k): 
        query_feats = self.index.reconstruct(id_query).reshape(1,-1) #lay embedding goc cua anh id_query 


        # Tim k anh gan nhat 
        scores, idx_image = self.index.search(query_feats, k=k)
        idx_image = idx_image.flatten()

        #lay dung duong dan cua anh 
        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info for info in infos_query]

        
        return scores, idx_image, infos_query, image_paths
    
    def text_search(self, text, k):
        if detect(text) == 'vi':
            text = self.translater(text) # --> ham nay giup chuyen tieng viet sang tieng anh 

        ###### TEXT FEATURES EXACTING ###### encode thanh embedding de truy van 
        text = clip.tokenize([text]).to(self.device) # --> Nhap vap CLIP model lay vector embedding cua cau 
        text_features = self.model.encode_text(text).cpu().detach().numpy().astype(np.float32)

        ###### SEARCHING ##### tim k anh gan dung nhat 
        scores, idx_image = self.index.search(text_features, k=k)
        idx_image = idx_image.flatten()

        ###### GET INFOS KEYFRAMES_ID ######
        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info for info in infos_query]

        return scores, idx_image, infos_query, image_paths