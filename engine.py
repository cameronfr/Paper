#religioussculptland_pix2pix uses a CycleGan trained on laplacian edge detector <-> full image as a better edge detectorself.
#The cyclegan is used to process wikiart images to get their edges. Then a pix2pix model is trained from these very simple edges -> full image.
#uses https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix -- run in same folder, or have IPython/Hydrogen running in same folder.
#If try edges from training set vs that same edge with half occluded, can see if model is overfitting.

# imports for flask part
import flask #use flask==0.12.2 and debug=False to run in ipython
from flask import Flask, request
from flask_cors import CORS
from PIL import Image, ImageFile
import numpy as np
import base64
import io
import os
import matplotlib.pyplot as plt
import logging
# imports for CycleGan part
from options.test_options import TestOptions
from models import create_model
import torch
import torchvision.transforms as transforms
# main
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)
CORS(app)
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

def setupModel():
    opt = TestOptions().parse() #get all cyclegan's options
    opt.model = "cycle_gan"
    opt.name = "hhs_to_drawing"
    # opt.name = "religioussculptland_cycle2"
    # opt.netG = "unet_256" #pix2pix
    # opt.no_lsgan = True
    # opt.norm = "batch"
    # opt.no_flip = True
    opt.display_id = -1
    opt.input_nc = 3
    opt.isTrain = False
    opt.eval = True
    opt.output_nc = 3
    opt.loadSize =256
    opt.fineSize =256
    model = create_model(opt)
    model.setup(opt)
    return model

model = setupModel()

def runModel(input):
    data = {'A': input, 'B':torch.randn(*input.shape), 'A_paths':[], 'B_paths':[]}
    model.set_input(data)
    model.test()
    output = model.fake_B
    return output

def preprocessSketchImg(img):
    img = img.convert('RGB')
    # img = img.convert('L')
    img = np.asarray(img)
    smallerSideLen = np.min(img.shape[:2])
    img = img[:smallerSideLen, :smallerSideLen]
    img = Image.fromarray(img)
    img = transforms.Resize((256,256))(img)
    img = transforms.ToTensor()(img)
    img = (img - 0.5) / 0.5
    img = img.unsqueeze(0)
    return img

# ------------------------------ FLASK ---- #

@app.route("/")
def index():
    return "Hello sketches"

img = Image.open("epoch008_real_A2.png")
@app.route('/image', methods=['POST'])
def login():
    if request.method == 'POST':
        #todo: current image size is 7kb according to chrome. If change to svg by constructing one from points, can maybe? reduce size
        data = request.data.split(b'base64,')[1]
        img = base64.decodebytes(data)
        img = Image.open(io.BytesIO(img))

        img = preprocessSketchImg(img)
        out = runModel(img)

        # plt.hist(img[0].cpu().numpy().flatten())
        # plt.show()
        plt.imshow(img[0].cpu().numpy().transpose(1, 2, 0) *0.5 + 0.5)
        # plt.show()
        plt.imshow(out[0].cpu().numpy().transpose(1, 2, 0) *0.5 + 0.5)
        # plt.show()
        out = ((out.cpu().numpy() *0.5) + 0.5) * 256
        out = out.squeeze(0).transpose(1, 2, 0).astype(np.uint8)
        # out = Image.fromarray(out.squeeze(2), mode='L') #pix2pix
        out = Image.fromarray(out)
        buffered = io.BytesIO()
        out.save(buffered, format="JPEG")
        out = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return "data:image/jpeg;base64," + out
    else:
        pass

app.run(host="0.0.0.0", port=8989, debug=False)
plt.show()
