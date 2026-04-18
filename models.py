### CNN as a precursor to the semantic segmentation model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


class ConfidenceMapCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder---downsamples(the convolutional filter covers a greater area on the image)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # Output confidence map
        self.outputs = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        # Decoder
        x = F.relu(self.upconv1(x))
        x = F.relu(self.conv7(x))

        x = F.relu(self.upconv2(x))
        x = F.relu(self.conv8(x))

        x = F.relu(self.upconv3(x))
        x = F.relu(self.conv9(x))

        ### Converts any raw value to a confidence score [0,1] ---2 classes only
        confidence_map = torch.sigmoid(self.outputs(x))
        return confidence_map

    def train_epoch(self, optimizer, loss_fn, train_dataloader, device):
        self.train() ## sets the object(data) in training mode
        running_loss = 0.0
        pbar = tqdm(train_dataloader, desc='Training')

        for image_input, target in pbar:
            # Move data to device
            image_input = image_input.to(device) ## so that data is on the same device as the model --easier for forward pass
            target = target.to(device)

            # set the optimiser's internal gradient to 0 before gradient descent algo
            optimizer.zero_grad()

            # Forward pass
            outputs = self(image_input)  ### directly outputs to the sigmoid function and confidence map

            # Calculate loss
            loss = loss_fn(outputs, target)

            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        train_loss = running_loss / len(train_dataloader)  ### Average training loss for 1 batch
        return train_loss

    # Validation function
    def validate(self, val_loader, criterion, device):
        self.eval()
        running_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for image_input, targets in pbar:
                image_input = image_input.to(device)
                targets = targets.to(device)

                outputs = self(image_input)

                # Calculate loss
                loss = criterion(outputs, targets)
                running_loss += loss.item()

                pbar.set_postfix({'loss': loss.item()})

            val_loss = running_loss / len(val_loader)
            return val_loss


loss_fn = nn.CrossEntropyLoss()
model = ConfidenceMapCNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

## Hyperparameters
num_epochs = 50
batch_size = [16,32,48,64] # This variable is not used in the current setup, dataloaders define batch size
lr = [0.1,0.01,0.001] # This variable is not used in the current setup, optimizer defines learning rate
best_val_loss = float('inf')
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 50)

    # Train
    train_loss = model.train_epoch(optimizer, loss_fn, train_dataloader, device)
    print(f'Train Loss: {train_loss:.4f}')

    # Validate (if you have val_dataloader)
    # The validate_dataloader is assumed to be defined in a previous cell and available in the global scope
    val_loss = model.validate(validate_dataloader, loss_fn, device)
    print(f'Val Loss: {val_loss:.4f}')
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_confidence_model.pth')
        print(f'Saved best segmentation model!')

print('Training completed!')

