import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test_joke, y_test_reaction, joke_classes, reaction_classes):
    y_pred_joke, y_pred_reaction = model.predict(X_test)
    y_pred_joke = np.argmax(y_pred_joke, axis=1)
    y_pred_reaction = np.argmax(y_pred_reaction, axis=1)

    print("Joke Type Classification Report:")
    print(classification_report(y_test_joke, y_pred_joke, target_names=joke_classes))
    print("\nReaction Type Classification Report:")
    print(classification_report(y_test_reaction, y_pred_reaction, target_names=reaction_classes))

    return y_pred_joke, y_pred_reaction

def get_confusion_matrices(y_test_joke, y_pred_joke, y_test_reaction, y_pred_reaction):
    cm_joke = confusion_matrix(y_test_joke, y_pred_joke)
    cm_reaction = confusion_matrix(y_test_reaction, y_pred_reaction)
    return cm_joke, cm_reaction