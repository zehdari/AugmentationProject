# !pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YQfqPBdBcd52L0qmUV7A")
project = rf.workspace("rfdetrcomputervision").project("football-players-detection-3zvbc-mbric")
version = project.version(1)
dataset = version.download("coco")
                