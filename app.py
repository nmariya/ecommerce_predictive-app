from flask import Flask, render_template, jsonify
import pickle
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

# @app.route('/preprocessing')
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    ps = PorterStemmer()
    tokens = [ps.stem(token) for token in tokens]
    return tokens
with open('train_dictionary.pk', 'r') as f:
    pickle.load(res2,f)
n_words = len(res2)
def wordToIndex(word):
    if word in res2:
        return res2.index(word)
    else:
        return -1
def wordToTensor(word):
    tensor = torch.zeros(1, n_words)
    if wordToIndex(word) != -1:
        tensor[0][wordToIndex(word)] = 1
        return tensor
    else:
        return tensor    
def lineToTensor(line):
    l= line.split()
    tensor = torch.zeros(len(l), 1, n_words)
    for li, word in enumerate(l):
        tensor[li][0][wordToIndex(word)] = 1
    return tensor

def load_image(path, shape):
        image = np.array(Image.open(path+'.jpg'))
        image = cv2.resize(image, (shape[0], shape[1]))
        image = np.divide(image, 255)
        return image 



# @app.route('/predict')
def evaluate(line_tensor):
    hidden_lstm = rnn_lstm.initHidden()
    for i in range(line_tensor.size()[0]):
        output_lstm, hidden_lstm = rnn_lstm(line_tensor[i], hidden_lstm)
    return output_lstm
def predict(input_line, n_predictions=1):
    with torch.no_grad():
        output_lstm = evaluate(lineToTensor(input_line))
    return output_lstm
INPUT_SHAPE = (299,299,3)
def image_predict(path, INPUT_SHAPE):
    image = load_image(path, INPUT_SHAPE)
    score_predict_t = model.predict(image[np.newaxis])[0]
    label_predict = np.argmax(score_predict)
    predicted.append(label_predict)


@app.route('/fruits')
def fruits():
    beers = [
        {
            'brand': 'Guinness',
            'type': 'stout'
        },
        {
            'brand': 'Hop House 13',
            'type': 'lager'
        }
    ]
    list_of_fruits = ['banana', 'orange', 'apple']
    list_of_drinks = ['coke', 'milk', beers]
    return jsonify(Fruits=list_of_fruits, Drinks=list_of_drinks)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')