import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():
    if "reset_app" not in st.session_state:
        st.session_state.reset_app = False

    st.title('ML Classifiers on the Iris Dataset')

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

    if "X_test" not in st.session_state: 
        st.session_state["X_Test"] = []
    
    if "y_test_pred" not in st.session_state: 
        st.session_state["y_test_pred"] = []

    if "selected_kernel" not in st.session_state: 
        st.session_state["selected_kernel"] = []

def display_form1():
    st.session_state["current_form"] = 1
    form1 = st.form("intro")

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    form1.text(text)

    form1.header('Description')
    form1.image('iris_flower.jpg', caption="Iris Plant", use_column_width=True)

    form1.subheader('The Iris Dataset')

    text = """The Iris dataset is a well-known and widely used dataset in the field
    of machine learning. Here's a breakdown of its key aspects:"""
    form1.write(text)
    text = """Data points: 150, representing 50 samples from each of three Iris species: 
    Iris setosa, Iris versicolor, and Iris virginica.
    \nFeatures: Four measurements for each flower (in centimeters): Sepal length, 
    Sepal width, Petal length, Petal width
    \nTarget variable: The species of the Iris flower (Setosa, Versicolor, or Virginica)."""
    form1.write(text)
    form1.write('Applications:')
    text = """Commonly used to introduce and test various machine learning algorithms, 
    especially for: Classification (predicting the flower species based on the
    measurements) Visualization (exploring relationships between features and species)"""
    form1.write(text)
    form1.write('Interesting characteristics:')
    text = """One species (Setosa) is easily distinguishable from the others 
    based on its sepal measurements. The other two species (Versicolor and Virginica) 
    have some overlap in their measurements, making them more challenging to 
    distinguish solely based on two features. Overall, the Iris dataset, despite its 
    simplicity, offers a valuable resource for understanding and practicing 
    fundamental machine learning concepts."""
    form1.write(text)

    submit1 = form1.form_submit_button("Start")

    if submit1:
        form1 = [];
        # Go to the next form        
        display_form2()

def display_form2():
    st.session_state["current_form"] = 2

    form2 = st.form("training")
    # Load the iris dataset
    data = load_iris()

    # Convert data features to a DataFrame
    feature_names = data.feature_names
    df = pd.DataFrame(data.data, columns=feature_names)
    df['target'] = data.target
    
    form2.write('The iris dataset')
    form2.write(df)

    # Separate features and target variable
    X = df.drop('target', axis=1)  # Target variable column name
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    text = """Imagine a triangle formed by the three centroids of class 0. 
    Class 1 data points will be scattered around the center of the triangle, 
    with some points falling inside the triangle and overlapping with 
    class 0 data points. This creates a challenging scenario for SVMs with linear kernels,
    as a straight line cannot perfectly separate the overlapping regions."""

    form2.write(text)

    form2.subheader('Browse the Dataset') 
    form2.write(df)

    form2.subheader('Dataset Description')
    form2.write(df.describe().T)

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a scatter plot with color based on species
    sns.scatterplot(
        x="sepal width (cm)",
        y="sepal length (cm)",
        hue="target",
        palette="deep",
        data=df,
        ax=ax,
    )

    # Add labels and title
    ax.set_xlabel("Sepal Width (cm)")
    ax.set_ylabel("Sepal Length (cm)")
    ax.set_title("Sepal Width vs. Sepal Length by Iris Species")

    # Add legend
    plt.legend(title="Species")

    # Show the plot
    form2.pyplot(fig)

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a scatter plot with color based on species
    sns.scatterplot(
        x="petal width (cm)",
        y="petal length (cm)",
        hue="target",
        palette="deep",
        data=df,
        ax=ax,
    )

    # Add labels and title
    ax.set_xlabel("Petal Width (cm)")
    ax.set_ylabel("Petal Length (cm)")
    ax.set_title("Petal Width vs. Petal Length by Iris Species")

    # Add legend
    plt.legend(title="Species")

    # Show the plot
    form2.pyplot(fig)

    form2.subheader('Select the kernel')

    # Create the selecton of classifier
    clf = SVC(kernel='linear')
    st.session_state['selected_kernel'] = 0
    options = ['Linear', 'Polynomial', 'Radial Basis Function']
    selected_option = form2.selectbox('Select the kernel', options)
    if selected_option =='Polynomial':
        st.session_state['selected_kernel'] = 1
        clf = SVC(kernel='poly', degree=3)
    elif selected_option=='Radial Basis Function':
        st.session_state['selected_kernel'] = 2
        clf = SVC(kernel='rbf', gamma=10) 
    else:
        clf = SVC(kernel='linear')
        st.session_state['selected_kernel'] = 0

    # save the clf to the session variable
    st.session_state['clf'] = clf

    submit2 = form2.form_submit_button("Train")
    if submit2:     

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

        clf = st.session_state['clf']
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)

        form2.subheader('Confusion Matrix')
        form2.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        form2.text(cm)
        form2.subheader('Performance Metrics')
        form2.text(classification_report(y_test, y_test_pred))

        # save the clf to the session state
        st.session_state['clf'] = clf
        display_form3()

        # save data to session state
        st.session_state['X_test'] = X_test
        st.session_state['y_test_pred'] = y_test_pred

def display_form3():
    st.session_state["current_form"] = 3
    form3 = st.form("Visualization")
    form3.subheader('Visualization')
    form3.text('Click the button to generate a plot of the data.')

    plotbn = form3.form_submit_button("Plot")
    if plotbn:                    

        clf = st.session_state['clf']
        X_test = st.session_state['X_test']
        y_test_pred = st.session_state['y_test_pred']
        visualize_classifier(form3, clf, X_test, y_test_pred)

        if st.session_state['selected_kernel'] == 0:
            text = """For partially overlapping clusters, the linear kernel might be
            able to find a hyperplane (straight line in higher dimensions) that 
            separates the majority of points, but misclassifications will 
            likely occur due to the overlap."""
        elif st.session_state['selected_kernel'] == 1:
            text = """The polynomial kernel can be more effective with 
            overlapping clusters compared to the linear kernel. By mapping the 
            data to a higher-dimensional space, it can potentially find non-linear 
            decision boundaries that better separate the classes even if 
            they overlap in the original feature space."""
        else:
            text = """The RBF kernel is often the most robust choice for dealing 
            with overlapping clusters. It uses a Gaussian function to measure 
            similarity between data points, allowing for flexible and smooth decision
            boundaries even in complex, non-linear scenarios."""
            
        form3.write(text)
    submit3 = form3.form_submit_button("Reset")
    if submit3:
        st.session_state.reset_app = True
        st.session_state.clear()

def visualize_classifier(form, classifier, X, y, title=''):
    # Define the minimum and maximum values for X and Y
    # that will be used in the mesh grid
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Define the step size to use in plotting the mesh grid 
    mesh_step_size = 0.01

    # Define the mesh grid of X and Y values
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    # Run the classifier on the mesh grid
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Reshape the output array
    output = output.reshape(x_vals.shape)
    
    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Specify the title
    ax.set_title(title)
    
    # Choose a color scheme for the plot
    ax.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    
    # Overlay the training points on the plot
    ax.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    
    # Specify the boundaries of the plot
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(y_vals.min(), y_vals.max())
    
    # Specify the ticks on the X and Y axes
    ax.set_xticks(np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0))
    ax.set_yticks(np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0))

    
    form.pyplot(fig)


if __name__ == "__main__":
    app()
