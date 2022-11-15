import json
import glob
from pathlib import Path
import re
import io
import zipfile
import tensorflow as tf

with open("config.json", "r") as f:
    config = json.load(f)

config


# Slice and tag
class Cbz2dataset:
    def __init__(self, config):
        self.__dict__.update(config)
        
        self.workspace = Path(self.workspace)
        self.orig_images = Path(self.orig_images)
        self.orig_annotations = Path(self.orig_annotations)
        
        self.files = self._load_file_list()
        
    def _stringlist_fixedlen(self, xs:list, zfill:bool):
        # make string length const along list
        maxlen = max(len(x) for x in xs)
        if zfill: 
            return [x.zfill(maxlen)[:maxlen] for x in xs]
        else:
            return [x.ljust(maxlen,"_")[:maxlen] for x in xs]
    
        
    def _load_file_list(self):
        files = [Path(x) for x in glob.glob((self.orig_images/"**").as_posix(), recursive=True)]
        dirs = [x.parent for x in files]
        dir_files = {k:[] for k in set(dirs)}
        for dirname, filename in zip(dirs,files):
            info = {
                "path":filename, 
                "type":filename.suffix.lower(),
                # Extracts number index from filename
                "index":re.sub('\D', '_', filename.stem).split("_")[-1],
            }
            dir_files[dirname].append(info)
        for k, v in dir_files.items():
            indices = [x["index"] for x in v]
            indices = self._stringlist_fixedlen(indices, zfill=True)
            indices = [x+ "_" + self._normalize_string(info["path"].name) for x,info in zip(indices, v)]
            indices = self._stringlist_fixedlen(indices, zfill=False)
            for info, x in zip(v, indices):
                info["index"] = x
        return dir_files
                
    def process(self):
        for x in self._process():
            print(x)
            raise NotImplementedError("TODO: self.dump")
            self.dump(x)
    
    def _load_image(self, info:dict):
        if info["type"] in (".zip",".cbz"):
            raise NotImplementedError("extension")
        elif info["type"] in (".png", ".jpg", ".jpeg", ".gif", ".bmp"):
            image = tf.io.read_file(info["path"].as_posix())
            image = tf.image.decode_image(image, channels=3)
            info['height'], info['width'], _ = image.shape
            info['offset_height'], info['offset_width'] = 0, 0
            info['data'] = image
        else:
            print(f"Unrecognized filetype : {info}")
        
        if self.split_half and (info["width"] > info["height"]) :
            info_left, info_right = info.copy(), info.copy()
            offset = info["width"]//2
            info_left['data'] = info['data'][:,:offset,:]
            info_right['data'] = info['data'][:,offset:,:]
            info_right['offset_width'] += offset
            infos = [info_right, info_left] if self.right_to_left else [info_left, info_right]
        else:
            infos = [info]
        
        for i, info in enumerate(infos):
            info['index'] += f"_{i}"[:2]
            yield info
    
                
    def _process(self):
        for dirname, filelist in self.files.items():
            for info in filelist:
                infos = self._load_image(info)
                for info in infos:
                    info = self._annotation(info)
                    info = self._segmentation(info)
                    info = self._resize(info)
                    yield info
                
    def process_cbz(self):
        raise NotImplementedError()
        zip_buffer = io.BytesIO()
        file_like_object = io.BytesIO(my_zip_data)
        zipfile_ob = zipfile.ZipFile(file_like_object)
    def _normalize_string(self, x:str):
        # TODO
        # Replace unicode string with / + hex code
        return x
    def _annotation(self, info:dict):
        # TODO
        return info
    def _segmentation(self, info:dict):
        # TODO
        return info
    def _resize(self, info:dict):
        # TODO
        return info

self=Cbz2dataset(config)
self.process()


