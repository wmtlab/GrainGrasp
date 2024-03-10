import json
from attrdict import AttrDict

cfgs = json.load(open("config.json"))
cfgs = AttrDict(cfgs)
