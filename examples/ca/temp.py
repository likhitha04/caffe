# set up Python environment: numpy for numerical routines, and matplotlib for plotting
from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import caffe
import datetime
import werkzeug
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
from decimal import Decimal, ROUND_HALF_EVEN
sys.path.append('/home/speech/darkflow-master')
from os.path import isfile, join
import time
from darkflow.net.build import  TFNet
import cv2
import scipy.misc

# display plots in this notebook
 #%matplotlib inline
UPLOAD_FOLDER = '/tmp/'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])
app = Flask(__name__) 

def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data

# set display defaults
@app.route('/')
def home():
    return render_template('1.html')

@app.route('/load',methods = ['POST', 'GET'])
def load():
        #f=request.files['imagefile']
        imagefile = request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)

        imagefile.save(filename)
        image = exifutil.open_oriented_im(filename)
        
        plt.rcParams['figure.figsize'] = (10, 10)        # large images
        plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels 
        plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.

        caffe_root = '/home/speech/.local/install/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
        sys.path.insert(0, caffe_root + 'python')

# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

        if os.path.isfile(caffe_root + 'models/placesCNN/places205CNN_iter_300000.caffemodel'):
            print ('CaffeNet found')
        else:
            print ('Downloading pre-trained CaffeNet model...')

   
        caffe.set_mode_cpu()

        model_def = caffe_root + 'models/placesCNN/places205CNN_deploy.prototxt'
        model_weights = caffe_root + 'models/placesCNN/places205CNN_iter_300000.caffemodel'

        net = caffe.Net(model_def,      # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
        mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
        mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
        print ('mean-subtracted values:', zip('BGR', mu))

# create transformer for the input called 'data'
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

        transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
        transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
        net.blobs['data'].reshape(50,        # batch size
                                 3,         # 3-channel (BGR) images
                                 227, 227)  # image size is 227x227


        img = caffe.io.load_image(filename)
        transformed_image = transformer.preprocess('data', img)
        plt.imshow(img)



# copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image

### perform classification
        output = net.forward()

        output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
        
      
        print ('predicted class is:', output_prob.argmax())

# load ImageNet labels
        labels_file = caffe_root + 'data/place/categoryIndex_places205.csv'

        labels = np.loadtxt(labels_file, str, delimiter='\t')

        
        #thresh = raw_input("Enter input: ")
        thresh = 0.1
        top_inds = output_prob.argsort()[::-1]
        items = []
        for i in top_inds:
                a = Decimal(str(output_prob[i])).quantize(Decimal('.01'), rounding=ROUND_HALF_EVEN)
                b = Decimal(str(thresh)).quantize(Decimal('.01'), rounding=ROUND_HALF_EVEN)
                if a.compare(b) >= 0:
                       an_item = dict(a1=output_prob[i],a2=labels[i])
                       items.append(an_item)

        
        
        options = {"model": "/home/speech/darkflow-master/cfg/tiny-yolo.cfg", "load": "/home/speech/darkflow-master/bin/tiny-yolo.weights", "threshold": 0.1}

        tfnet = TFNet(options)
        
        imgcv = cv2.imread(filename)
        result = tfnet.return_predict(imgcv)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for r in result:
           cv2.rectangle(imgcv,
			(r['topleft']['x'], r['topleft']['y']), (r['bottomright']['x'], r['bottomright']['y']),(0,255,0),1)
           cv2.putText(imgcv, r['label'], (r['topleft']['x'], r['topleft']['y']+12),font,0.47,(255,0,0),1)


        cv2.imwrite('outfile.jpg', imgcv)
        files='/home/speech/.local/install/caffe/outfile.jpg'
        i = exifutil.open_oriented_im(files)

        return render_template('2.html',imagesrc=embed_image_html(image),items=items,result=result,isrc=embed_image_html(i))

if __name__ == '__main__':
  app.run(debug = True)
 
