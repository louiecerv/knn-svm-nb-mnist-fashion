import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():
    if "reset_app" not in st.session_state:
        st.session_state.reset_app = False

    text = """Image Classification on the MNIST Fashion Dataset"""
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

    form1.image('fashion.png', caption="Modified National Institute of Standards and Technology", use_column_width=True)
    text = """The Fashion MNIST dataset is a popular choice for testing and comparing machine
    learning algorithms, particularly those suited for image classification. 
    \nRelatively small size: With 70,000 images, it's computationally efficient to train and 
    test on, making it ideal for initial experimentation and algorithm evaluation.
    \nSimple image format: The images are grayscale and low-resolution (28x28 pixels), 
    simplifying preprocessing and reducing computational demands.
    \nMultiple classes: It consists of 10 distinct clothing categories, allowing you to assess 
    the classifiers' ability to differentiate between various categories.
    \nBenchmarking: As a widely used dataset, it facilitates comparison of your models' 
    performance with established benchmarks for these algorithms on the same dataset."""
    form1.write(text)
  
    submit1 = form1.form_submit_button("Start")

    if submit1:
        # Go to the next form        
        display_form2()

def display_form2():
    st.session_state["current_form"] = 2

    form2 = st.form("training")

    # Download and load the Fashion MNIST dataset
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)
    
    # Extract only the specified number of images and labels
    size = 10000
    X = X[:size]
    y = y[:size]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #save the values to the session state    
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test

    form2.write("""The MNIST Fashion classification task is an image 
        classification problem where the goal is to automatically categorize 
        images of different clothing items.""")
    text = """1. K-Nearest Neighbors (KNN):
        \nConcept: KNN classifies an image by comparing it to its k nearest neighbors in the 
        training data. The class label of the majority of these neighbors becomes the predicted 
        label for the new image.
        \nPerformance: KNN can achieve good accuracy on the Fashion-MNIST dataset, 
        especially with careful selection of the "k" parameter (number of neighbors). 
        However, it can be computationally expensive for large datasets like this one due to 
        the need for distance calculations with all training data points during prediction.
        \n2. Support Vector Machine (SVM):
        \nConcept: SVM aims to find a hyperplane in the feature space that best 
        separates the data points of different classes. It maximizes the margin between 
        the hyperplane and the closest data points of each class (support vectors).
        \nPerformance: SVMs generally perform well on the Fashion-MNIST dataset, 
        offering good accuracy and handling high-dimensional data efficiently. However, 
        tuning the hyperparameters of an SVM can be challenging, and it might not be as 
        interpretable as other models like KNN.
        \n3. Naive Bayes: 
        \nConcept: Naive Bayes assumes independence between features and uses Bayes' 
        theorem to calculate the probability of an image belonging to each class 
        based on its individual pixel values.
        \nPerformance: Naive Bayes can be a fast and efficient classifier for the 
        Fashion-MNIST dataset. However, its assumption of feature independence can be a 
        limitation, leading to potential inaccuracies when features are not truly 
        independent, as often seen in image data."""
    form2.write(text)

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']

    form2.subheader('First 25 images in the MNIST fashion dataset') 

    # Get the first 25 images and reshape them to 28x28 pixels
    train_images = np.array(X_train)
    train_labels = np.array(y_train)
    images = train_images[:25].reshape(-1, 28, 28)
    # Create a 5x5 grid of subplots
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    # Plot each image on a separate subplot
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images[i], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Class: {train_labels[i]}")
    # Show the plot
    plt.tight_layout()
    form2.pyplot(fig)

    form2.subheader('Select the classifier')

    # Create the selecton of classifier

    clf = KNeighborsClassifier(n_neighbors=5)
    options = ['K Nearest Neighbor', 'Support Vector Machine', 'Naive Bayes']
    selected_option = form2.selectbox('Select the classifier', options)
    if selected_option =='Support Vector Machine':
        clf = SVC(kernel='rbf')
        st.session_state['selected_model'] = 1
    elif selected_option=='Naive Bayes':        
        clf = GaussianNB()
        st.session_state['selected_model'] = 2
    else:
        clf = KNeighborsClassifier(n_neighbors=5)
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
    if st.session_state['selected_model'] == 0:     # KNN
        text = """KNN achieves good accuracy on the Fashion MNIST dataset, often 
        reaching around 85-90%. However, it can be slow for large datasets 
        due to needing to compare each test image to all training images. 
        Additionally, choosing the optimal number of neighbors (k) can be 
        crucial for performance."""
        classifier = 'K-Nearest Neighbor'
    elif st.session_state['selected_model'] == 1:   # SVM
        text = """SVM can also achieve high accuracy on this dataset, 
        similar to KNN. It offers advantages like being memory-efficient, 
        but choosing the right kernel function and its parameters 
        can be challenging."""
        classifier = 'Support Vector Machine'
    else:   #Naive Bayes
        text = """Naive Bayes is generally faster than the other two options but 
        may achieve slightly lower accuracy, typically around 80-85%. It performs 
        well when the features are independent, which might not perfectly hold true 
        for image data like the Fashion MNIST."""
        classifier = "Naive Bayes"
    

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
