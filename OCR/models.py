import torch
import torch.nn as nn
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class ResidualBlock(nn.Module):
    def __init__(self, planes):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class ResidualNet(nn.Module):
    def __init__(self, input_planes, height, width, number_of_blocks, classes):
        super(ResidualNet, self).__init__()

        if number_of_blocks < 2:
            raise ValueError("The residual net needs at least two blocks.")

        self.conv1 = conv3x3(input_planes, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.residual1 = ResidualBlock(16)
        self.conv2 = conv3x3(16, 32, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.residual2 = ResidualBlock(32)
        self.conv3 = conv3x3(32, 64, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.later_residual_blocks = nn.ModuleList()

        for _ in range(number_of_blocks-2):
            self.later_residual_blocks.append(ResidualBlock(64))

        self.dense_input_dim = height * width * 4
        self.dense = nn.Linear(self.dense_input_dim, classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.residual1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.residual2(x)
        x = self.relu(self.bn3(self.conv3(x)))

        for block in self.later_residual_blocks:
            x = block(x)

        return self.dense(x.view(-1, self.dense_input_dim))

class CNN_Sequence_Extractor(nn.Module):
    def __init__(self, nchannels, leakyRelu=False):
        super(CNN_Sequence_Extractor, self).__init__()

        # Size of the kernel (image filter) for each convolutional layer.
        ks = [3, 3, 3, 3, 3, 3, 2]
        # Amount of zero-padding for each convoutional layer.
        ps = [1, 1, 1, 1, 1, 1, 0]
        # The stride for each convolutional layer. The list elements are of the form (height stride, width stride).
        ss = [(2,2), (2,2), (1,1), (2,1), (1,1), (2,1), (1,1)]
        # Number of channels in each convolutional layer.
        nm = [64, 128, 256, 256, 512, 512, 512]

        # Initializing the container for the modules that make up the neural network the neurel netowrk.
        cnn = nn.Sequential()

        # Represents a convolutional layer. The input paramter i signals that this is the ith convolutional layer. The user also has the option to set batchNormalization to True which will perform a batch normalization on the image after it has undergone a convoltuional pass. There is no output but this function adds the convolutional layer module created here to the sequential container, cnn.
        def convRelu(i, batchNormalization=False):
            nIn = nchannels if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('leaky_relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # Creating the 7 convolutional layers for the model.
        convRelu(0)
        convRelu(1)
        convRelu(2, True)
        convRelu(3)
        convRelu(4, True)
        convRelu(5)
        convRelu(6, True)

        self.cnn = cnn

    def forward(self, input, widths=None):
        output = self.cnn(input)
        _, _, h, _ = output.size()
        assert h == 1, "the height of conv must be 1"
        output = output.squeeze(2) # [b, c, w]
        output = output.permute(2, 0, 1) #[w, b, c]

        if widths is not None:
          sorted_widths, idx = widths.sort(descending=True)
          output = output.index_select(1, idx)
          output = nn.utils.pack_padded_sequence(output, sorted_widths / 4)

        return output

class CRNN(nn.Module):
    def __init__(self, nchannels, nclass, nhidden, num_layers=2, leakyRelu=False):
        super(CRNN, self).__init__()

        # Instantiating the convolutional and recurrent neural net layers as attributes of the CRNN module
        self.cnn = CNN_Sequence_Extractor(nchannels, leakyRelu)
        self.rnn = nn.LSTM(512, nhidden, num_layers, bidirectional=True)
        self.embedding = nn.Linear(nhidden * 2, nclass)

    # A forward pass through the CRNN. Takes a batch of images as input and produces a tensor corresponding to vertical slices of the image x batch size x predicted probability of membership to each class.
    def forward(self, input, widths=None):
        # conv features
        conv = self.cnn(input, widths=widths)

        # A forward pass through the LSTM layers. Takes in a batch of inputs and passes them through the LSTM layers.
        recurrent, _ = self.rnn(conv)

        if widths is not None:
          recurrent = nn.utils.rnn.pad_packed_sequence(recurrent)

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class ImageEncoder(nn.Module):
  def __init__(self, nchannels, nhidden, num_layers, attention=True):
    self.core = nn.Sequential([CNN_Sequence_Extractor(nc), nn.LSTM(512, nhidden, num_layers, bidirectional=True)])
    self.register_buffer('reverse_indices', torch.LongTensor(range(1, num_layers*2, 2)))


  def forward(self, input, widths=None):
    output, (all_hiddens, _) = self.core(input, widths=widths)

    if widths is not None:
      output = nn.utils.rnn.pad_packed_sequence(output)

    reverse_hiddens = all_hiddens.index_select(0, Variable(reverse_indices))

    if attention:
      return output, reverse_hiddens
    else:
      return reverse_hiddens

class MultilayerLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_layers, bias=True):
        super(MultilayerLSTMCell, self).__init__()
        self.lstm_layers = nn.ModuleList()

        if isinstance(hidden_sizes, int):
            temp = [input_size]

            for _ in range(num_layers):
                temp.append(hidden_sizes)

            hidden_sizes = temp
        else:
            hidden_sizes = [input_size] + hidden_sizes
        for i in range(num_layers):
            curr_lstm = nn.LSTMCell(hidden_sizes[i], hidden_sizes[i+1], bias=bias)
            self.lstm_layers.append(curr_lstm)

    def forward(self, input, hiddens, cell_states):
        result_hiddens, result_cell_states = [], []
        curr_input = input

        for lstm_cell, curr_hidden, curr_cell_state in zip(self.lstm_layers, hiddens, cell_states):
            curr_input, new_cell_state = lstm_cell(curr_input, (curr_hidden, curr_cell_state))
            result_hiddens.append(curr_input)
            result_cell_states.append(new_cell_state)

        return torch.stack(result_hiddens), torch.stack(result_cell_states)

class Sequence_to_Sequence_Model(nn.Module):
    """
      For the decoder this expects something like an lstm cell and not an lstm.
      The number of layers in the encoder and decoder should be the same. Also the hidden
      sizes should be the same.
    """
    def __init__(self, encoder, decoder, hidden_size, nclass, embedding_size):
      super(Sequence_to_Sequence_Model, self).__init__()
      self.encoder = encoder
      self.decoder = decoder
      # nclass + 2 to include end of sequence and trash
      self.output_log_odds = nn.Linear(hidden_size, nclass+2)
      self.extract_initial_hidden = nn.Linear(hidden_size, hidden_size)
      self.softmax = nn.Softmax(dim=0)

      self.register_buffer('SOS_token', torch.LongTensor([[nclass+2]]))
      self.EOS_value = nclass + 1

      # nclass + 3 to include start of sequence, end of sequence, and trash.
      # n + 2 - start of sequence, end of sequence - n + 1, trash - n.
      # The first n correspond to the alphabet in order.
      self.embedding = nn.Embedding(nclass+3, embedding_size)

      #nclass + 1 is the trash category to avoid penalties after target's EOS token
      self.loss_func = nn.CrossEntropyLoss(ignore_index=nclass)

    """
        input: The output of the encoder for the input should correspond to the hidden state and cell state of the
         1st time step for the reverse part of the model. They should be [num_layers, batch_size, hidden_size].
        target: The target should have dimensions, (seq_len x batch_size), and should be a LongTensor.
    """
    def forward_train(self, input, target, teacher_forcing=False):
      decoder_hiddens, decoder_cell_states = self.encoder(input)
      target_length, batch_size = target.size()
      SOS_token = Variable(self.SOS_token)
      decoder_input = self.embedding(SOS_token).squeeze(0).repeat(batch_size, 1) # batch_size x embedding_size

      loss = 0

      for i in range(target_length):
        decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, (decoder_hiddens, decoder_cell_states))

        decoder_hidden = decoder_hiddens[-1]
        log_odds = self.output_log_odds(decoder_hidden)
        loss += self.loss_func(log_odds, target[i])

        if teacher_forcing:
          next_input = target[i].unsqueeze(0)
        else:
          _, next_input = log_odds.topk(1)

        decoder_input = self.embedding(next_input).squeeze(0) # batch x embedding size

      return loss

    """
      Inputs must be of batch size 1. Point wise prediction with a batch size greater than 1 is possible, but messy.
      The main issue is if the batch consists of inputs with very different lengths for their label, than a lot
      of wasted computation will occur. Whether it is worth it to do batching anyway is a mystery left for the
      future. It returns a list of numbers corresponding to the labels it guessed.
    """
    def point_wise_prediction(self, input, maximum_length=20):
      decoder_hiddens, decoder_cell_states = self.encoder(input)
      SOS_token = Variable(self.SOS_token)
      decoder_input = self.embedding(SOS_token).squeeze(0) # batch_size x embedding_size
      output_so_far = []

      for _ in range(maximum_length):
        decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, (decoder_hiddens, decoder_cell_states))
        decoder_hidden = decoder_hiddens[-1]
        log_odds = self.output_log_odds(decoder_hidden)

        _, next_input = log_odds.topk(1)

        if int(next_input) == self.EOS_value:
          break

        output_so_far.append(int(next_input))
        decoder_input = self.embedding(next_input).squeeze(0) # batch x embedding size

      return output_so_far

    """
      Similar to point wise prediction this is restricted to batch size 1. This one would be a good deal harder to make work with batches
      and unlike point prediction I don't even see any good way to have this one work with batches.
    """
    def beam_search_prediction(self, input, maximum_length=20, beam_width=5):
      decoder_hiddens, decoder_cell_states = self.encoder(input)
      SOS_token = Variable(self.SOS_token)
      decoder_input = self.embedding(SOS_token).squeeze(0) # batch_size x embedding_size
      word_inputs = []

      for _ in range(beam_width):
        word_inputs.append((0, [], True, [decoder_input, decoder_hiddens, decoder_cell_states]))

      for _ in range(maximum_length):
        new_word_inputs = []

        for i in range(beam_width):
            if not word_inputs[i][2]:
                continue

            decoder_input, decoder_hiddens, decoder_cell_states = word_inputs[i][3]
            decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, (decoder_hiddens, decoder_cell_states))

            decoder_hidden = decoder_hiddens[-1]
            log_odds = self.output_log_odds(decoder_hidden).squeeze(0) # nclasses
            log_probs = self.softmax(log_odds).log_()

            value, next_input = log_probs.topk(beam_width) # beam_width, beam_width
            log_value = value.log()

            decoder_input = self.embedding(next_input.unsqueeze(1)) # beam_width x batch_size x embedding size

            new_word_inputs.extend((word_inputs[i][0] + float(log_value[k]), word_inputs[i][1].append(int(index[k])),
                                    int(index[k]) == self.EOS_value, [decoder_input[k], decoder_hiddens, decoder_cell_states])
                                    for k in range(beam_width))
        word_inputs = sorted(new_word_inputs, key=lambda word_input: word_input[0])[-beam_width:]
      return word_inputs[-1][1]


class Sequence_to_Sequence_Attention_Model(Sequence_to_Sequence_Model):
    def __init__(self, encoder, decoder, hidden_size, nclass, embedding_size,
                 alignment_size):
      super(Sequence_to_Sequence_Attention_Model, self).__init__(encoder, decoder, hidden_size, nclass)

      self.attention_hidden = nn.Linear(2 * hidden_size, alignment_size)
      self.attention_context = nn.Linear(hidden_size, alignment_size, bias=False)
      self.tanh = nn.Tanh()
      self.attention_alignment_vector = nn.Linear(encoded_size, 1)

    """
        input: The output of the encoder for the input should be a pair. The first part of the pair should
               correspond to the annotations. It should be [seq_len, batch_size, hidden_size * 2]. The
               second part of the pair should correspond to the hidden state of the 1st time step for
               the reverse part of the model. It should be [num_layers, batch_size, hidden_size].
        target: The target should have dimensions, (seq_len x batch_size), and should be a LongTensor.
    """
    def forward_train(self, input, target, teacher_forcing=False):
      annotations, decoder_hiddens, decoder_cell_states = self.encoder(input)
      attention_hidden_values = self.attention_hidden(annotations) # seq_len x batch_size x alignment_size

      target_length, batch_size = target.size()
      num_layers, _, _ = decoder_hiddens.size()
      SOS_token = Variable(self.SOS_token)
      word_input = self.embedding(SOS_token).squeeze(0).repeat(batch_size, 1) # batch_size x embedding_size
      loss = 0

      for i in range(target_length):
        attention_logits = self.attention_alignment_vector(self.tanh(self.attention_context(decoder_hiddens[0]).unsqueeze(0)  + attention_hidden_values))
        attention_probs = self.softmax(attention_logits) # seq_len x batch_size x 1
        context_vec = (attention_probs * annotations).sum(0) # batch_size x hidden_size * 2
        decoder_input = torch.cat((word_input, context_vec), dim=1)

        decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, (decoder_hiddens, decoder_cell_states))
        decoder_hidden = decoder_hiddens[-1]
        log_odds = self.output_log_odds(decoder_hidden)
        loss += self.loss_func(log_odds, target[i])

        if teacher_forcing:
          next_input = target[i].unsqueeze(0)
        else:
          _, next_input = log_odds.topk(1)

        word_input = self.embedding(next_input).squeeze(0) # batch x embedding size

      return loss


    """
      Inputs must be of batch size 1. Point wise prediction with a batch size greater than 1 is possible, but messy.
      The main issue is if the batch consists of inputs with very different lengths for their label, than a lot
      of wasted computation will occur. Whether it is worth it to do batching anyway is a mystery left for the
      future. It returns a list of numbers corresponding to the labels it guessed.
    """
    def point_wise_prediction(self, input, maximum_length=20):
      annotations, decoder_hiddens, decoder_cell_states = self.encoder(input)

      attention_hidden_values = self.attention_hidden(annotations) # seq_len x batch_size x alignment_size
      num_layers, _, _ = decoder_hiddens.size()
      SOS_token = Variable(self.SOS_token)
      word_input = self.embedding(SOS_token).squeeze(0) # batch_size x embedding_size
      output_so_far = []

      for _ in range(maximum_length):
        attention_logits = self.attention_alignment_vector(self.tanh(self.attention_context(decoder_hiddens[0]).unsqueeze(0)  + attention_hidden_values))
        attention_probs = self.softmax(attention_logits) # seq_len x batch_size x 1
        context_vec = (attention_probs * annotations).sum(0) # batch_size x hidden_size * 2
        decoder_input = torch.cat((word_input, context_vec), dim=1)

        decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, (decoder_hiddens, decoder_cell_states))
        decoder_hidden = decoder_hiddens[-1]
        log_odds = self.output_log_odds(decoder_hidden)
        _, next_input = log_odds.topk(1)

        if int(next_input) == self.EOS_value:
          break

        output_so_far.append(int(next_input))
        word_input = self.embedding(next_input).squeeze(0) # batch x embedding size

      return output_so_far

    """
      Similar to point wise prediction this is restricted to batch size 1. This one would be a good deal harder to make work with batches
      and unlike point prediction I don't even see any good way to have this one work with batches.
    """
    def beam_search_prediction(self, input, maximum_length=20):
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input)

        attention_hidden_values = self.attention_hidden(annotations) # seq_len x batch_size x alignment_size
        num_layers, _, _ = decoder_hiddens.size()
        SOS_token = Variable(self.SOS_token)
        word_input = self.embedding(SOS_token).squeeze(0) # batch_size x embedding_size

        word_inputs = []

        for _ in range(beam_width):
            word_inputs.append((0, [], True, [word_input, decoder_hiddens, decoder_cell_states]))

        for _ in range(maximum_length):
            new_word_inputs = []

            for i in range(beam_width):
                if not word_inputs[i][2]:
                    continue

                word_input, decoder_hiddens, decoder_cell_states = word_inputs[i][3]

                attention_logits = self.attention_alignment_vector(self.tanh(self.attention_context(decoder_hiddens[0]).unsqueeze(0)  + attention_hidden_values))
                attention_probs = self.softmax(attention_logits) # seq_len x batch_size x 1
                context_vec = (attention_probs * annotations).sum(0) # batch_size x hidden_size * 2
                decoder_input = torch.cat((word_input, context_vec), dim=1)

                decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, (decoder_hiddens, decoder_cell_states))
                decoder_hidden = decoder_hiddens[-1]
                log_odds = self.output_log_odds(decoder_hidden).squeeze(0) # nclasses
                log_probs = self.softmax(log_odds).log_()

                value, next_input = log_probs.topk(beam_width) # beam_width, beam_width
                log_value = value.log()

                word_input = self.embedding(next_input.unsqueeze(1)) # beam_width x batch_size x embedding size

                new_word_inputs.extend((word_inputs[i][0] + float(log_value[k]), word_inputs[i][1].append(int(index[k])),
                                        int(index[k]) == self.EOS_value, [word_input[k], decoder_hiddens, decoder_cell_states])
                                        for k in range(beam_width))
            word_inputs = sorted(new_word_inputs, key=lambda word_input: word_input[0])[-beam_width:]

        return word_inputs[-1][1]
