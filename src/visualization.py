import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import shap
import numpy as np

def plot_confusion_matrices(y_test_joke, y_pred_joke, y_test_reaction, y_pred_reaction, le_joke, le_reaction):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    sns.heatmap(confusion_matrix(y_test_joke, y_pred_joke), annot=True, fmt='d', xticklabels=le_joke.classes_, yticklabels=le_joke.classes_)
    plt.title('Confusion Matrix - Joke Type')
    plt.subplot(122)
    sns.heatmap(confusion_matrix(y_test_reaction, y_pred_reaction), annot=True, fmt='d', xticklabels=le_reaction.classes_, yticklabels=le_reaction.classes_)
    plt.title('Confusion Matrix - Reaction Type')
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(history.history['joke_output_accuracy'], label='Train')
    plt.plot(history.history['val_joke_output_accuracy'], label='Validation')
    plt.title('Joke Type Accuracy')
    plt.legend()
    plt.subplot(122)
    plt.plot(history.history['reaction_output_accuracy'], label='Train')
    plt.plot(history.history['val_reaction_output_accuracy'], label='Validation')
    plt.title('Reaction Type Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_shap_values(model, X_train, X_test):
    explainer = shap.DeepExplainer(model, X_train[:100])
    shap_values = explainer.shap_values([X_test[:10], X_test[:10]])

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    shap.summary_plot(shap_values[0][0], X_test, plot_type="bar")
    plt.title('SHAP Values - Joke Type')
    plt.subplot(122)
    shap.summary_plot(shap_values[1][0], X_test, plot_type="bar")
    plt.title('SHAP Values - Reaction Type')
    plt.tight_layout()
    plt.show()

    return shap_values

def plot_psychological_importance(shap_values, feature_types):
    feature_importance_joke = np.mean(np.abs(shap_values[0][0]), axis=0)
    feature_importance_reaction = np.mean(np.abs(shap_values[1][0]), axis=0)

    psychological_importance = {
        'Joke Type': {},
        'Reaction Type': {}
    }

    start = 0
    for feature_name, feature_length in feature_types:
        end = start + feature_length
        psychological_importance['Joke Type'][feature_name] = np.mean(feature_importance_joke[start:end])
        psychological_importance['Reaction Type'][feature_name] = np.mean(feature_importance_reaction[start:end])
        start = end

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.bar(psychological_importance['Joke Type'].keys(), psychological_importance['Joke Type'].values())
    plt.title('Psychological Feature Importance - Joke Type')
    plt.ylabel('Mean |SHAP Value|')
    plt.subplot(122)
    plt.bar(psychological_importance['Reaction Type'].keys(), psychological_importance['Reaction Type'].values())
    plt.title('Psychological Feature Importance - Reaction Type')
    plt.ylabel('Mean |SHAP Value|')
    plt.tight_layout()
    plt.show()

    return psychological_importance

def print_classification_reports(y_test_joke, y_pred_joke, y_test_reaction, y_pred_reaction, le_joke, le_reaction):
    print("Joke Type Classification Report:")
    print(classification_report(y_test_joke, y_pred_joke, target_names=le_joke.classes_))
    print("\nReaction Type Classification Report:")
    print(classification_report(y_test_reaction, y_pred_reaction, target_names=le_reaction.classes_))

def interpret_results(psychological_importance, correlations):
    print("\nCross-modal Analysis Interpretation:")
    for i, corr in enumerate(correlations, 1):
        print(f"Component {i}: Correlation = {corr:.4f}")
        if corr > 0.5:
            print(f"  Strong correlation between audio and visual features in component {i}")
        elif corr > 0.3:
            print(f"  Moderate correlation between audio and visual features in component {i}")
        else:
            print(f"  Weak correlation between audio and visual features in component {i}")

    print("\nOverall Cross-modal Insights:")
    print("This analysis reveals the extent to which audio and visual features are related")
    print("in the context of humor perception. Strong correlations suggest that certain")
    print("aspects of joke delivery are consistently expressed across both modalities,")
    print("while weak correlations might indicate unique information carried by each modality.")

    print("\nPsychological Interpretation:")
    print("1. Joke Type Prediction:")
    for feature, importance in psychological_importance['Joke Type'].items():
        print(f"   - {feature}: {importance:.4f}")
        if feature == 'Visual' and importance > 0.1:
            print("     This suggests that visual cues play a significant role in joke delivery.")
        elif feature == 'Facial' and importance > 0.1:
            print("     Facial expressions seem to be important in conveying the type of joke.")
        elif feature == 'Audio' and importance > 0.1:
            print("     The audio features, including tone and pitch, are crucial in joke classification.")
        elif feature == 'Motion' and importance > 0.1:
            print("     Body language and gestures contribute to the joke type.")
        elif feature == 'Interaction' and importance > 0.1:
            print("     The physical interaction between parent and child influences the joke type.")
        elif feature == 'Temporal' and importance > 0.1:
            print("     The timing and duration of the joke delivery are significant factors.")

    print("\n2. Reaction Type Prediction:")
    for feature, importance in psychological_importance['Reaction Type'].items():
        print(f"   - {feature}: {importance:.4f}")
        if feature == 'Visual' and importance > 0.1:
            print("     Visual cues are important in determining the child's reaction.")
        elif feature == 'Facial' and importance > 0.1:
            print("     The child's facial expressions are key indicators of their reaction.")
        elif feature == 'Audio' and importance > 0.1:
            print("     Auditory responses (e.g., laughter, verbal reactions) play a significant role.")
        elif feature == 'Motion' and importance > 0.1:
            print("     The child's body language is an important factor in their reaction.")
        elif feature == 'Interaction' and importance > 0.1:
            print("     The physical interaction between parent and child influences the reaction.")
        elif feature == 'Temporal' and importance > 0.1:
            print("     The timing of the reaction relative to the joke is a significant factor.")

    print("\nOverall Psychological Insights:")
    print("This analysis provides a quantitative measure of the relative importance of different")
    print("psychological and sensory factors in humor perception and reaction among children.")
    print("It highlights the multimodal nature of humor, showing how visual, auditory, and")
    print("interactive elements contribute to both joke delivery and reception.")
    print("The cross-modal analysis further reveals how these different modalities interact,")
    print("providing a more comprehensive understanding of the humor perception process.")