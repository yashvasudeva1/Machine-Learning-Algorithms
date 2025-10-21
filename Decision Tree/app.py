import streamlit as st
import joblib
from io import StringIO
data=joblib.load('Decision Tree/Decision_Tree.pkl')
model=data['model']
classification_report=data['Classification Report']
accuracy_score=data['Accuracy Score']
confusion_matrix=data['Confusion Matrix']
st.title('Decision Tree Classifier')
tab1,tab2,tab3=st.tabs(['Unhyperparameterized','Hyperparameterized with Postpruning', 'Hyperparameterized with Prepruning'])
with tab1:
    st.header('Unhyperparameterized')
    sepal_length=st.number_input('sepal_length',0.0,10.0)
    sepal_width=st.number_input('sepal_width',0.0,10.0)
    petal_length=st.number_input('petal_length',0.0,10.0)
    petal_width=st.number_input('petal_width',0.0,10.0)
    if st.button('Predict'):
        prediction=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
        if prediction[0]==0:
            st.write('Setosa')
        elif prediction[0]==1:
            st.write('Versicolor')
        else:
            st.write('Virginica')
    if st.button('View Report'):
        st.write("Hello")
            


















