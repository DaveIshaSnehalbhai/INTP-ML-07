import flask
import pickle
import torch
import numpy as np
from torch import nn

app = flask.Flask(__name__)   

class model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
        nn.Linear(10,200),
        nn.ReLU(),
        nn.Linear(200,300),
        nn.ReLU(),
        nn.Linear(300,400),
        nn.ReLU(),
        nn.Linear(400,500),
        nn.ReLU(),
        nn.Linear(500,600),
        nn.ReLU(),
        nn.Linear(600,700),
        nn.ReLU(),
        nn.Linear(700,800),
        nn.ReLU(),
        nn.Linear(800,900),
        nn.ReLU(),
        nn.Linear(900,1000),
        nn.ReLU(),
        nn.Linear(1000,1000),
        nn.ReLU(),
        nn.Linear(1000,1000),
        nn.ReLU(),
        nn.Linear(1000,1000),
        nn.ReLU(),
        nn.Linear(1000,900),
        nn.ReLU(),
        nn.Linear(900,800),
        nn.ReLU(),
        nn.Linear(800,700),
        nn.ReLU(),
        nn.Linear(700,600),
        nn.ReLU(),
        nn.Linear(600,500),
        nn.ReLU(),
        nn.Linear(500,400),
        nn.ReLU(),
        nn.Linear(400,300),
        nn.ReLU(),
        nn.Linear(300,200),
        nn.ReLU(),
        nn.Linear(200,100),
        nn.ReLU(),
        nn.Linear(100,1))
    
    def forward(self,x):
        return self.layer(x)
        
    

@app.route('/', methods=['GET'])

def home():        

    return flask.render_template('index.html')


@app.route('/', methods=['POST'])   

def predict():   

    if(flask.request.method == 'POST'):

        cycle = int(flask.request.form['c1'])

        LPC_outlet_temperature= float(flask.request.form['c2'])

        LPT_outlet_temperature = float(flask.request.form['c3'])

        HPC_outlet_pressure= float(flask.request.form['c4'])

        HPC_outlet_Static_pressure = float(flask.request.form['c5'])

        Ratio_of_fuel_flow_to_Ps30 = float(flask.request.form['c6'])

        Bypass_Ratio = float(flask.request.form['c7'])

        Bleed_Enthalpy = float(flask.request.form['c8'])

        High_pressure = float(flask.request.form['c9'])

        Low_pressure = float(flask.request.form['c10'])
        

    model = model_1()
    model.load_state_dict(torch.load('model_1.pth',map_location='cpu'),strict=False)
            

    x = [cycle,LPC_outlet_temperature,LPT_outlet_temperature, HPC_outlet_pressure, HPC_outlet_Static_pressure ,Ratio_of_fuel_flow_to_Ps30, Bypass_Ratio,Bleed_Enthalpy,High_pressure,Low_pressure]
    x = torch.Tensor(x)

    model.eval()
    with torch.inference_mode():
        answer = model(x)
    answer = answer.numpy()
    print(x)
    return flask.render_template('index.html', prediction = answer)


if __name__ == '__main__':

    app.run(debug=True)