import torch
import torch.nn as nn
import torch.nn.functional as F

class Contraction(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
    )

  def forward(self, x):
    return self.conv(x)

class Expansion(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.expansion = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
    self.conv = Contraction(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.expansion(x1)
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)

class Unet(nn.Module):
  def __init__(self, n_channels, n_classes, conv_dims=128):
    super(Unet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.conv_dims = conv_dims
    #Encoder Path
    self.in_conv = Contraction(n_channels, self.conv_dims)
    self.con1 = Contraction(self.conv_dims, self.conv_dims*2)
    self.con2 = Contraction(self.conv_dims*2, self.conv_dims*4)
    self.con3 = Contraction(self.conv_dims*4, self.conv_dims*8)
    # Bottleneck
    self.con4 = Contraction(self.conv_dims*8, self.conv_dims*16)
    #Decoder Path
    self.exp1 = Expansion(self.conv_dims*16, self.conv_dims*8)
    self.exp2 = Expansion(self.conv_dims*8, self.conv_dims*4)
    self.exp3 = Expansion(self.conv_dims*4, self.conv_dims)
    #Final Layer
    self.out_conv = nn.Conv2d(self.conv_dims, n_classes, kernel_size=1)
    # Optimizer and Loss
    self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
    self.loss = nn.MSEloss()

  def forward(self, x):
    x1 = self.in_conv(x)
    x2 = F.max_pool2d(x1, 2)
    x3 = F.max_pool2d(self.con1(x2), 2)
    x4 = F.max_pool2d(self.con2(x3), 2)
    x5 = F.max_pool2d(self.con3(x4), 2)
    x6 = self.con4(F.max_pool2d(x5, 2))

    x = self.exp1(x6, x5)
    x = self.exp2(x, x4)
    x = self.exp3(x,x3)
    logits = self.out_conv(x)
    return logits
  # Train - Function

  def train_model(self, train_loader, num_epochs=25):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            self.optim.zero_grad()  # Zero the parameter gradients
            outputs = self.forward(inputs)  # Forward pass
            loss = self.loss(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            self.optim.step()  # Optimize
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

class TimeShiftedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, time_shift=0):
        super(TimeShiftedConv2D, self).__init__()
        self.time_shift = time_shift
        # Ensure the in_channels here matches the number of channels in the input tensor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.time_shift != 0:
            x = torch.roll(x, shifts=self.time_shift, dims=-1)
            if self.time_shift > 0:
                x[..., :self.time_shift] = 0
            else:
                x[..., self.time_shift:] = 0
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class UNetWithTimeShift(nn.Module):
    def __init__(self):
        super(UNetWithTimeShift, self).__init__()
        self.conv_dims=128
        # Adjust in_channels for the first layer to match the input tensor's channel size
        self.in_conv = TimeShiftedConv2D(1, self.conv_dims, (3, 3), padding=1, time_shift=2) # Assuming the input has 1 channel
        self.con2 = TimeShiftedConv2D(self.conv_dims, self.conv_dims*2, (3, 3), padding=1, time_shift=2)
        self.con3 = TimeShiftedConv2D(self.conv_dims*2, self.conv_dims*4, (3, 3), padding=1, time_shift=2)
        self.con4 = TimeShiftedConv2D(self.conv_dims*4, self.conv_dims*8, (3, 3), padding=1, time_shift=2)

        self.pool = nn.MaxPool2d((2, 2))

        self.exp1 = nn.ConvTranspose2d(self.conv_dims*8, self.conv_dims*4, 2, stride=2)
        self.exp1_conv1 = TimeShiftedConv2D(self.conv_dims*8, self.conv_dims*4, (3, 3), padding=1, time_shift=-2)
        self.exp2 = nn.ConvTranspose2d(self.conv_dims*4, self.conv_dims*2, 2, stride=2)
        self.exp2_conv2 = TimeShiftedConv2D(self.conv_dims*4, self.conv_dims*2, (3, 3), padding=1, time_shift=-2)
        self.exp3 = nn.ConvTranspose2d(self.conv_dims*2, self.conv_dims, 2, stride=2)
        self.exp3_conv3 = TimeShiftedConv2D(self.conv_dims*2, self.conv_dims, (3, 3), padding=1, time_shift=-2)

        self.final = nn.Conv2d(self.conv_dims, 1, kernel_size=(1, 1))

        # Optimizer and Loss
        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss = nn.MSELoss()


    def forward(self, x):
        # Encoding path
        d1 = self.in_conv(x)
        p1 = self.pool(d1)
        d2 = self.con2(p1)
        p2 = self.pool(d2)
        d3 = self.con3(p2)
        p3 = self.pool(d3)
        d4 = self.con4(p3)

        # Decoding path + skip connections
        u3 = self.exp1(d4)
        # Resize or crop d3 to match u3's size if they don't match
        d3 = self.match_size(d3, u3)
        u3 = torch.cat((u3, d3), dim=1)
        u3 = self.exp1_conv1(u3)

        u2 = self.exp2(u3)
        # Resize or crop d2 to match u2's size
        d2 = self.match_size(d2, u2)
        u2 = torch.cat((u2, d2), dim=1)
        u2 = self.exp2_conv2(u2)

        u1 = self.exp3(u2)
        # Resize or crop d1 to match u1's size
        d1 = self.match_size(d1, u1)
        u1 = torch.cat((u1, d1), dim=1)
        u1 = self.exp3_conv3(u1)

        out = self.final(u1)
        return out

    def match_size(self, tensor_to_resize, reference_tensor):
        """
        Resize or crop the tensor_to_resize to match the spatial dimensions of the reference_tensor.
        """
        _, _, h_ref, w_ref = reference_tensor.size()
        _, _, h, w = tensor_to_resize.size()

        # If the height and width match, no need to resize or crop
        if h == h_ref and w == w_ref:
            return tensor_to_resize

        # Center crop tensor_to_resize to the size of reference_tensor
        crop_h = (h - h_ref) // 2
        crop_w = (w - w_ref) // 2

        # Adjusting for odd differences by adding 1 to the end if necessary
        return tensor_to_resize[:, :, crop_h:crop_h+h_ref, crop_w:crop_w+w_ref]
    
    def train_model(self, train_loader, num_epochs=25):
      for epoch in range(num_epochs):
          running_loss = 0.0
          for inputs, targets in train_loader:
              self.optim.zero_grad()  # Zero the parameter gradients
              outputs = self.forward(inputs)  # Forward pass
              loss = self.loss(outputs, targets)  # Compute loss
              loss.backward()  # Backward pass
              self.optim.step()  # Optimize
              running_loss += loss.item() * inputs.size(0)
          epoch_loss = running_loss / len(train_loader.dataset)
          print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')