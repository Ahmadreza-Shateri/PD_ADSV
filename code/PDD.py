import numpy as np
import pandas as pd
import gradio as gr
from tensorflow.keras.preprocessing import image


from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

data = pd.read_csv('NewDataset.csv')
x = data.drop(['Status'], axis=1)
y = data['Status']


model1 = lgb.LGBMClassifier()
model2 = XGBClassifier(eta=0.001, gamma=0, n_estimators=94)
model5= GradientBoostingClassifier()
model9 = BaggingClassifier(XGBClassifier(eta=0.001, gamma=0, n_estimators=94))
model = VotingClassifier(estimators=[('lgb', model1),('xgb', model2), ('gb', model5), ('bc', model9)],weights=(6,13,6,1), voting='hard')
model.fit(x, y)

def parkinson(csv_file):

   dataset = pd.read_csv(csv_file.name, delimiter=',')
   dataset.fillna(0, inplace=True)
   X = dataset.iloc[:, :-1]

   prediction = model.predict(X)
   
   if prediction == 1:
      return  image.load_img('Positive (2).jpg'), "This patient has Parkinson's Disease"
      
   else:
      return  image.load_img('Negative (2).jpg'), "There is no sign of disease in this patient"



with gr.Blocks(css="#btn0 {background: #4E16A8} #img0, #img1 {background:#0B0F19}") as app:
    gr.Markdown(
    """
    # Diagnosis of Parkinson's Disease
    ‏‏‎ ‎
    """)
    with gr.Row() as row:
        with gr.Column():
            img1 = gr.Image('Ribbon.svg', show_label=False, visible=False)
        with gr.Column():
            img1 = gr.Image('Ribbon.svg', show_label=False, elem_id="img0", invert_colors=False).style(full_width=True, height=135)
        with gr.Column():
            img1 = gr.Image('Ribbon.svg', show_label=False, visible=False)
    with gr.Row() as row:
        with gr.Column():
            img1 = gr.Image('Ribbon.svg', show_label=False, visible=False)
        with gr.Column():
            inpt = gr.inputs.File(label='CSV file')
            with gr.Row() as row:
                with gr.Column():
                    output2 = gr.Image(show_label=False, elem_id="img1").style(full_width=True, height=290)
            output1 = gr.outputs.Textbox(label='Result:')
            btn = gr.Button("Submit", elem_id="btn0")
        with gr.Column():
            img1 = gr.Image('Ribbon.svg', show_label=False, visible=False)
                
    btn.click(fn=parkinson, inputs=inpt, outputs=[output2, output1])
    with gr.Column():
        examples = gr.Examples(examples=[["POSITIVE case.csv"], ["NEGATIVE case.csv"]],
                           inputs=[inpt])

app.launch(share= True, inline=True)