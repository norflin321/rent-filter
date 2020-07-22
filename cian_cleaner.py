from bs4 import BeautifulSoup
import requests
import webbrowser
import os


def load_flat_dicts(url):
    cian_html = requests.get(url).text
    soup = BeautifulSoup(cian_html, "html.parser")

    flats_dict = {}
    flats = soup.find_all('div', {'data-name': 'TopOfferCard'})
    flats += soup.find_all('div', {'data-name': 'OfferCard'})

    for f in flats:
        flat_imgs = []

        additional_imgs = f.find_all('img', {'data-name': 'GalleryImage'})
        try:
            flat_imgs.append(f.find('img')['src'])
        except:
            continue 

        for fa in additional_imgs:
            flat_imgs.append(fa['src'])

        links = f.find_all('a', {'target': '_blank'})
        for a in links:
            if 'https://www.cian.ru/rent/flat/' or 'https://www.cian.ru/sale/flat/' in a['href']:
                if '/cat.php?' not in a['href']:
                  flats_dict[a['href']] = flat_imgs
                  break



    return flats_dict

test_url = "https://www.cian.ru/cat.php?currency=2&deal_type=rent&engine_version=2&maxprice=25000&minprice=15000&offer_type=flat&region=1&room1=1&room2=1&type=4"
flats_dict = load_flat_dicts(test_url)
print('Testing done!')

# Commented out IPython magic to ensure Python compatibility.
# %%time
#2. Building model and predict one image

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import efficientnet.tfkeras as efn
import io
from urllib.request import urlopen

#0.96
def build_model_efficientnet():
    pretrained_model = efn.EfficientNetB0(weights='imagenet', include_top=False)
    pretrained_model.trainable = False
    x = pretrained_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=pretrained_model.input, outputs=predictions)

    # lr=1e-4
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


### LOADING MODEL
model = build_model_efficientnet()
model = load_model('./cian_model_01_B0.h5')
print('Model Loaded!')


def _fast_expand(img):
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def pseudo_download_image(url):
    #     print(f'[INFO] Downloading {url}')
   resp = urlopen(url)
   image = np.asarray(bytearray(resp.read()), dtype="uint8")
   image = cv2.imdecode(image, cv2.IMREAD_COLOR)

   return image



def predict_image(url):
    img_size = 320
    #     open_cv_image = cv2.imread(img_path)
    open_cv_image = pseudo_download_image(url)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(open_cv_image, (img_size, img_size))
    orig_image = _fast_expand(test_image)
    result_orig = model.predict(orig_image, batch_size=1)

    #     classes = ['bad', 'good']
    # result_idx = np.argmax(result_orig)
    result_val = list(result_orig[0])

    return result_val

#Testing prediction
testing_res = predict_image('https://cdn-p.cian.site/images/8/766/147/kvartira-moskva-lomonosovskiy-prospekt-741667887-4.jpg')
print(f'Tested value: {testing_res}')

RESULTS = {}

def show_results(scores, threshold=0.5):
    print("Results:")
    count = 0
    html_file = open("results.html", "w")
    for f, scr in scores.items():
        img_thumb = scr['thumb']
        f_score = scr['score']
        class_idx = f_score.index(max(f_score))
        if class_idx == 1 and max(f_score) >= threshold:
            count += 1
            print(f, f_score)
            html_file.write(f'''
                <div style="text-align:center">
                    <span style="display:block; margin-top:20px">{count} | rate: {round(max(f_score), 2)}</span>
                    <a href="{f}"><img src='{img_thumb}' width=400/></a>
                </div>
                <hr>
            ''')

    html_file.close()
    webbrowser.open('file://' + os.path.realpath("results.html"))
    print("treshold:", threshold, " results:", count)


#3. Predict for all flats from result page
from IPython.core.display import display, HTML

def FilterAllShitPlease(cian_url, threshold=0.5):
    print(cian_url)
    CIAN_URL = cian_url    
    flats_dict = load_flat_dicts(cian_url)

    for f, urls in flats_dict.items():
        data = {'score': [0, 0], 'thumb': urls[0]}
        total_score = [0, 0]
        for url in urls:
            data['score'] = predict_image(url)
            total_score = [a + b for a, b in zip(total_score, data['score'])]

        final_score = [x/len(urls) for x in total_score]
        data['score'] = final_score
        RESULTS[f] = data
        classes = ['bad', 'good']
        print(f'Score for {f} -> {final_score}: This is a {classes[final_score.index(max(final_score))]} flat.')

  

THRESHOLD = 0.66
BASE_URL = "https://nn.cian.ru/cat.php"
#  for page in range(48):
    #  if page == 0: continue
    #  FilterAllShitPlease(f"{BASE_URL}?currency=2&deal_type=rent&engine_version=2&maxprice=20000&minfloor=3&offer_type=flat&p={page}&region=4885&type=4", threshold=THRESHOLD)


for page in range(19):
    if page == 0: continue
    FilterAllShitPlease(f"{BASE_URL}?currency=2&deal_type=rent&engine_version=2&maxprice=30000&minfloor=3&minprice=20000&offer_type=flat&p={page}&region=4885&type=4", threshold=THRESHOLD)

#ПОКАЗАТЬ ТОЛЬКО ОТСЕЯННЫЕ КВАРТИРЫ. threshold – порог качество от 0 до 1
show_results(RESULTS, threshold=THRESHOLD)
