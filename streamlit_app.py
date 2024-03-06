import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():
    if "reset_app" not in st.session_state:
        st.session_state.reset_app = False

    text = """Decision Tree, Random Forest and K-Nearest Neighbor on the MNIST Dataset"""
    st.subheader(text)

    # Use session state to track the current form
    if "current_form" not in st.session_state:
        st.session_state["current_form"] = 1    

    # Display the appropriate form based on the current form state
    if st.session_state["current_form"] == 1:
        display_form1()
    elif st.session_state["current_form"] == 2:
        display_form2()
    elif st.session_state["current_form"] == 3:
        display_form3()

    if "clf" not in st.session_state: 
        st.session_state["clf"] = []

    if "X_train" not in st.session_state: 
        st.session_state["X_train"] = []

    if "X_test" not in st.session_state: 
        st.session_state["X_test"] = []
    
    if "y_train" not in st.session_state: 
        st.session_state["X_train"] = []
    
    if "y_test" not in st.session_state: 
        st.session_state["y_yest"] = []

    if "selected_model" not in st.session_state: 
        st.session_state["selected_model"] = 0

def display_form1():
    st.session_state["current_form"] = 1
    form1 = st.form("intro")

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    form1.text(text)

    form1.image('MNIST.png', caption="Modified National Institute of Standards and Technology", use_column_width=True)
    text = """MNIST is a large database of handwritten digits that is commonly used for training and
    testing various image processing systems1234. The acronym stands for Modified National Institute 
    of Standards and Technology23. MNIST is a popular dataset in the field of machine learning and 
    can provide a baseline for benchmarking algorithms"""
    form1.write(text)
  
    submit1 = form1.form_submit_button("Start")

    if submit1:
        # Go to the next form        
        display_form2()

def display_form2():
    st.session_state["current_form"] = 2

    form2 = st.form("training")
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, data_home=".")

    # Separate features (data) and labels (target)
    size = 500
    X_train, X_test = mnist.data[:size], mnist.data[60000:]
    y_train, y_test = mnist.target[:size], mnist.target[60000:]

    #save the values to the session state    
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test

    form2.text('The task: Classify handwritten digits from 0 to 9 based on a given image.')
    text = """Dataset: MNIST - 70,000 images of handwritten digits (28x28 pixels), each labeled 
    with its corresponding digit (0-9).
    \nModels:
    \nK-Nearest Neighbors (KNN):
    \nEach image is represented as a 784-dimensional vector (28x28 pixels). 
    To classify a new image, its distance is measured to K nearest neighbors in the 
    training data. The majority class label among the neighbors is assigned to the new image.
    \nDecision Tree:
    \nA tree-like structure is built based on features (pixel intensities) of the images. 
    \nThe tree splits the data based on decision rules (e.g., "pixel intensity at 
    position X is greater than Y"). The new image is navigated through the tree based on 
    its features, reaching a leaf node representing the predicted digit class.
    \nRandom Forest:
    \nAn ensemble of multiple decision trees are built, each trained on a random subset of 
    features (pixels) and a random subset of data.
    \nTo classify a new image, it is passed through each decision tree, and the majority class 
    label from all predictions is assigned."""
    form2.write(text)

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']

    form2.subheader('First 25 images in the MNIST dataset') 

    # Get the first 25 images and reshape them to 28x28 pixels
    train_images = np.array(X_train)
    train_labels = np.array(y_train)
    images = train_images[:25].reshape(-1, 28, 28)
    # Create a 5x5 grid of subplots
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    # Plot each image on a separate subplot
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images[i], cmap=plt.cm.binary)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Digit: {train_labels[i]}")
    # Show the plot
    plt.tight_layout()
    form2.pyplot(fig)

    form2.subheader('Select the classifier')

    # Create the selecton of classifier

    clf = tree.DecisionTreeClassifier()
    options = ['Decision Tree', 'Random Forest Classifier', 'Extreme Random Forest Classifier', 'K Nearest Neighbor']
    selected_option = form2.selectbox('Select the classifier', options)
    if selected_option =='Random Forest Classifier':
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
        st.session_state['selected_model'] = 1
    elif selected_option=='Extreme Random Forest Classifier':        
        clf = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=0)
        st.session_state['selected_model'] = 2
    elif selected_option == 'K Nearest Neighbor':
        clf = KNeighborsClassifier(n_neighbors=5)
        st.session_state['selected_model'] = 3
    else:
        clf = tree.DecisionTreeClassifier()
        st.session_state['selected_model'] = 0

    # save the clf to the session variable
    st.session_state['clf'] = clf

    submit2 = form2.form_submit_button("Train")
    if submit2:     
        display_form3()

def display_form3():
    st.session_state["current_form"] = 3
    form3 = st.form("Result")
    classifier = ''
    if st.session_state['selected_model'] == 0:     # decision tree
        text = """Achieves good accuracy, but can be prone to 
        overfitting, leading to lower performance on unseen data.
        Simple and interpretable, allowing visualization of decision rules.
        Susceptible to changes in the training data, potentially 
        leading to high variance in predictions."""
        classifier = 'Decision Tree'
    elif st.session_state['selected_model'] == 1:   # Random Forest
        text = """Generally outperforms a single decision tree, 
        reaching accuracy close to 98%. Reduces overfitting through 
        averaging predictions from multiple trees. Ensemble method - 
        combines predictions from multiple decision trees, leading to 
        improved generalization and reduced variance. Less interpretable 
        compared to a single decision tree due to the complex 
        ensemble structure."""
        classifier = 'Random Forest'
    elif st.session_state['selected_model'] == 2:   # Extreme Random Forest
        text = """Performance: Can achieve similar or slightly better 
        accuracy compared to a random forest, but results can vary 
        depending on hyperparameter tuning. Introduces additional randomness 
        during tree building by randomly selecting features at each split.  Aims to 
        further improve generalization and reduce overfitting by increasing 
        the diversity of trees in the ensemble. Requires careful 
        hyperparameter tuning to achieve optimal performance."""
        classifier = "Extreme Random Forest"
    else:
        text = """K-Nearest Neighbor"""
        classifier = "K-Nearest Neighbor"

    form3.subheader('Performance of the ' + classifier)

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train= st.session_state['y_train']
    y_test = st.session_state['y_test']

    clf = st.session_state['clf']
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    form3.subheader('Confusion Matrix')
    form3.write('Confusion Matrix')
    cm = confusion_matrix(y_test, y_test_pred)
    form3.text(cm)

    form3.subheader('Performance Metrics')
    form3.text(classification_report(y_test, y_test_pred))

    form3.write(text)

    # save the clf to the session state
    st.session_state['clf'] = clf

    submit3 = form3.form_submit_button("Reset")
    if submit3:
        st.session_state.reset_app = True
        st.session_state.clear()
        form3.write("If the form does not reset, click the reset button again.")

if __name__ == "__main__":
    app()
