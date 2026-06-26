import torch.nn as nn
import torch.optim as optim
from modelling import get_data, preprocess, model, train, test
import dash
from interface import get_layout

def main():
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
    output = scores + trainHist.get_output()

def run_interface():
    app = dash.Dash(__name__)
    app.layout = get_layout()
    app.run(port=8050, debug=True)

if __name__ == "__main__":
    #main()
    run_interface()
