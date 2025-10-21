import streamlit as st
import joblib
from io import StringIO
data=joblib.load('Decision Tree/Decision_Tree.pkl')
model=data['model']
classificationreport=data['classificationreport']
accuracyscore=data['accuracyscore']
confusionmatrix=data['confusionmatrix']
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
        st.write(prediction)
        if prediction[0]==0:
            st.write('Setosa')
        elif prediction[0]==1:
            st.write('Versicolor')
        else:
            st.write('Virginica')
    if st.button('View Report'):
        col1,col2=st.columns(2)
        with col1:
          st.image()
        with col2:
          model.classification_report(y_test,y_pred)
          st.write(classification_report(y_test,y_pred))
          st.write(confusion_matrix(y_test,y_pred))
          st.write(accuracy_score(y_test,y_pred))
        if st.download_button(label="Download Metrics as Text File",
          data=text_content,
          file_name="classification_report.txt",
          mime="text/plain"):
          buffer = StringIO()
          buffer.write("Decision Tree Classification Report\n")
          buffer.write("="*40 + "\n\n")
          buffer.write(report)
          buffer.write("\n" + "="*40 + "\n")
          buffer.write("End of report\n")
          text_content = buffer.getvalue()








