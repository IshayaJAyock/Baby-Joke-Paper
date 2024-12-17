import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

def perform_cross_modal_analysis(visual_features, audio_features, n_components=5):
    cca = CCA(n_components=n_components)
    cca.fit(visual_features, audio_features)
    visual_cca, audio_cca = cca.transform(visual_features, audio_features)
    
    correlations = [np.corrcoef(visual_cca[:, i], audio_cca[:, i])[0, 1] for i in range(n_components)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_components + 1), correlations)
    plt.title('Cross-modal Correlations')
    plt.xlabel('CCA Component')
    plt.ylabel('Correlation')
    plt.show()
    
    return correlations

def interpret_cross_modal_analysis(correlations):
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