from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

class SimpleCNN(Model):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Define the layers in the __init__ method
        self.conv1 = Conv2D(16, (3, 3), padding='same', activation='relu')
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2))
        
        self.conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2))
        
        self.flatten = Flatten()
        self.fc = Dense(10)
        
    def call(self, inputs):
        # Define the forward pass in the call method
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Instantiate the model
model = SimpleCNN()
# Assuming an input shape of (64, 64, 3)
x = Input(shape=(64, 64, 3))
y = model(x)
