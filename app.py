import os
from flask          import (
        Flask, flash, render_template, request, redirect, session, url_for
        )
from werkzeug.utils import secure_filename
import bcrypt, jwt

import numpy as np
import json
from PIL import Image

import torch
from torchvision import models, transforms

HOME_FOLDER = os.getcwd()
UPLOAD_FOLDER = HOME_FOLDER+'/uploads/'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__ )
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'this is not a secret key'
app.users = {
        'testid':bcrypt.hashpw(b"testpassword",bcrypt.gensalt()),
        'testid2':bcrypt.hashpw(b"testpassword2",bcrypt.gensalt())        
        }

net = models.vgg16()
print('loading model...')
#load_weights = torch.load(HOME_FOLDER+'/vgg16-397923af.pth')
#net.load_state_dict(load_weights)
print('load complete')
net.eval()


class BaseTransform():

    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])

    def __call__(self,img):
        return self.base_transform(img)

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = BaseTransform(resize, mean, std)


indexsavepath = os.path.join(HOME_FOLDER+'/imagenet_class_index.json')

class ILSVRCPredictor():
    
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]

        return predicted_label_name

ILSVRC_class_index = json.load(open(indexsavepath,'r'))
predictor = ILSVRCPredictor(ILSVRC_class_index)


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods = ['GET'])
def index():
    #print(session)
    #print(request.cookies)
    username = request.cookies.get('username',"")
    if username in session:
        return f'hello, {username}!'
    return 'hello!', 200

@app.route('/login', methods = ['GET','POST'])
def login():
    if request.method == 'POST':
        
        username = request.form['username']
        password = request.form['password']
        
        if username in session:
            return f'You are already logged in, {username}' 
        
        if bcrypt.checkpw(password.encode('UTF-8'), app.users[username]):
            session[username] = True
            session.permanent = True
            response = redirect(url_for('upload_file'))
            response.set_cookie('username',username)
            return response

        else:
            return "", 401
          
    else:        
        return '''
            <form method="post">
                <p><input type=text name=username>
                <p><input type=password name=password>
                <p><input type=submit value=Login>
            </form>
        '''

@app.route('/upload', methods = ['GET','POST'])
def upload_file():
    username = request.cookies.get('username',"")
    if request.method == 'POST':
        result = ''
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(filepath)
            img = Image.open(filepath)
            img_transformed = transform(img)

            inputs = img_transformed.unsqueeze(0)
            out = net(inputs)
            result = predictor.predict_max(out)

            if os.path.exists(filepath):
                os.remove(filepath)
        else:
            flash('Allowed file type: png, jpg, jpeg, gif')
        return f'This is an image of {result}', 200
    if username in session and username=='testid':
        return render_template('upload.html')
    else:
        return 'You are not testid', 401

@app.route('/logout')
def logout():
    username = request.cookies['username']
    session.pop(username,None)
    print('logout',session)
    return redirect(url_for('index'))
