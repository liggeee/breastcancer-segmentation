import os
from flask import Flask, render_template,request,flash,redirect
import time
import numpy as np
from keras.models import load_model
from PIL import Image
from predict import predict

app = Flask(__name__, template_folder='templates',static_folder='templates/images')
app.config['SECRET_KEY']='secretkey'
UPLOAD_FOLDER = 'templates/images'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


size=128

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET','POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            extention = file.filename.rsplit('.', 1)[1].lower()
            filename = str(time.time())+'.'+extention
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            p = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img, mask_img = predict(p)
            mask_image='mask-'+filename
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            mask_img.save(os.path.join(app.config['UPLOAD_FOLDER'], mask_image))
            data = {
                'image' : filename,
                'mask_image' : mask_image   
            }
            return render_template('index.html',data=data)    
    return render_template('index.html')
