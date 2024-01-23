import numpy as np
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('6HK.html', form_values={})

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [float(x) for x in request.form.values() ]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    # Lưu lại các giá trị đã nhập
    form_values = request.form.to_dict()

    output = round(prediction[0])
    if output == 1:
        return render_template('6HK.html', prediction_text=f'Chúc mừng bạn ! Rất có thể bạn sẽ tốt nghiệp đúng hạn.', form_values=form_values)
    if output == 0:
        return render_template('6HK.html', prediction_text=f'Có thể bạn sẽ tốt nghiệp không đúng hạn. Hãy cố gắng hơn trong học tập nhé ', form_values=form_values)

if __name__ == "__main__":
    app.run(debug=True)