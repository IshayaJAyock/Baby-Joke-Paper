from src.model import create_model

def train_model(X_train, y_train_joke, y_train_reaction, input_shape, num_classes_joke, num_classes_reaction, batch_size, epochs, validation_split):
    model = create_model(input_shape, num_classes_joke, num_classes_reaction)
    
    history = model.fit(
        X_train, 
        [y_train_joke, y_train_reaction], 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=validation_split,
        verbose=1
    )
    
    return model, history