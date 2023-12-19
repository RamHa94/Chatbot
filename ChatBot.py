import torch
import torch.nn as nn

class SimpleBot(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleBot, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.word_to_ix = {}
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded.view(len(input), 1, -1))
        prediction = self.fc(hidden.squeeze(0))
        return prediction

    def add_word(self, word):
        if word not in self.word_to_ix:
            self.word_to_ix[word] = self.vocab_size
            self.vocab_size += 1
            new_embedding = torch.randn(1, self.embedding_dim)
            self.embedding.weight = nn.Parameter(torch.cat((self.embedding.weight, new_embedding), dim=0))

    def get_word_index(self, word):
        if word not in self.word_to_ix:
            return None
        return self.word_to_ix[word]

# Example usage
questions: list[str] = ["What is your name?", "How are you feeling today?", "What do you like to do?"]
vocab = set(" ".join(questions).lower().split())
word_to_ix: dict[str, int] = {word: i for i, word in enumerate(vocab)}
model = SimpleBot(len(vocab), 50, 100, 1)
model.word_to_ix = word_to_ix
model.vocab_size = len(vocab)

input_str = "What is your name?"
input_words: list[LiteralString] = input_str.lower().split()
for word in input_words:
    model.add_word(word)
input = torch.LongTensor([model.get_word_index(w) for w in input_words])
output = model(input)
print(output)