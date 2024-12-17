from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, num_classes_joke, num_classes_reaction):
    inputs = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output_joke = Dense(num_classes_joke, activation='softmax', name='joke_output')(x)
    output_reaction = Dense(num_classes_reaction, activation='softmax', name='reaction_output')(x)

    model = Model(inputs=inputs, outputs=[output_joke, output_reaction])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train_joke, y_train_reaction):
    history = model.fit(
        X_train, 
        [y_train_joke, y_train_reaction], 
        epochs=50, 
        batch_size=32, 
        validation_split=0.2,
        verbose=1
    )
    return history