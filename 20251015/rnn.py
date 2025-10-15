import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            if x.is_cuda:
                hidden = hidden.cuda()

        rnn_out, hidden = self.rnn(x, hidden)
        output = self.fc(rnn_out)
        
        return output, hidden
    
    def predict_next(self, x, hidden=None):
        output, hidden = self.forward(x, hidden)
        prediction = output[:, -1, :]
        return prediction, hidden
    
if __name__ == "__main__":
    print("=" * 50)
    print("test rnn model")
    print("=" * 50)

    model = SimpleRNN(input_size=1, hidden_size=32, num_layers=1, output_size=1)
    print("\nmodel structure:")
    print(model)

    batch_size=2
    sequence_length = 10
    input_size = 1

    test_input = torch.randn(batch_size, sequence_length, input_size)
    print(f"\nOutput data shape")

