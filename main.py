import torch
import torch.nn as nn
import torch.optim as optim
from util import get_device
from modelling import get_data, preprocess, model, train, test
import dash
from interface import get_layout
from pathlib import Path

def create_model():
    print("get_data")
    loaders = get_data.get_data()

    print("preprocess_data")
    preprocess.preprocess_data(loaders)

    print("get_model")
    _model = model.get_model()
    optimizer = optim.Adam(_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("train_model")
    trainHist = train.train_model(
        loaders,
        _model,
        optimizer,
        criterion
    )
    print("test_model")
    scores = test.test_model(trainHist, _model, loaders)

    torch.save(_model.state_dict(), "model/model.pth")
    output = scores + trainHist.get_output()

    with open("assets/output.txt", "w") as f:
        f.write(output)

def run_app(_model):
    app = dash.Dash(__name__)
    app.layout = get_layout(app, _model)
    app.run(port=8050, debug=True)

if __name__ == "__main__":

    weights = Path("model/model.pth")
    _model = model.get_model()

    if not weights.is_file():
        create_model()

    _model.load_state_dict(
        torch.load(weights, map_location=get_device())
    )
    _model.to(get_device())
    _model.eval()
    run_app(_model)
