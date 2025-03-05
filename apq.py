import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas
df= pd.read_csv('iris.csv')

df = df.dropna()
df.head()

## Part 1 - Data Preparation


# @title species vs island

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
plt.subplots(figsize=(8, 8))
df_2dhist = pd.DataFrame({
    x_label: grp['SepalLengthCm'].value_counts()
    for x_label, grp in df.groupby('Species')
})
sns.heatmap(df_2dhist, cmap='viridis')
plt.xlabel('Species')
_ = plt.ylabel('SepalLength')

# @title species

from matplotlib import pyplot as plt
import seaborn as sns
df.groupby('Species').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

sns.scatterplot(x='SepalLengthCm',y='SepalWidthCm',data=df,hue='Species',palette='Dark2')

sns.catplot(x='Species',y='PetalLengthCm',data=df,kind='box',palette='Dark2')

pd.get_dummies(df)

X = pd.get_dummies(df.drop('Species',axis=1),drop_first=True)
y = df['Species']

X.shape

y.shape

X.head()

y.head()

y.unique()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy')


model.fit(X_train, y_train)

from sklearn.tree import plot_tree
plt.figure(figsize=(12,8))
plot_tree(model);

plt.figure(figsize=(12, 8), dpi=150)
plot_tree(model, filled=True, feature_names=X.columns.tolist())  # Convert to list
plt.show()


base_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report, ConfusionMatrixDisplay

cm = confusion_matrix(y_test,base_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

disp.plot()

print(classification_report(y_test,base_pred))

model.feature_importances_

pd.DataFrame(index=X.columns,data=model.feature_importances_,columns=['Feature Importance'])

sns.boxplot(x='Species',y='PetalWidthCm',data=df)

def report_model(model):
    model_preds = model.predict(X_test)
    print(classification_report(y_test,model_preds))
    print('\n')
    plt.figure(figsize=(12,8),dpi=150)
    plot_tree(model,filled=True,feature_names=X.columns);

pruned_tree = DecisionTreeClassifier(criterion='entropy',max_depth=2)
pruned_tree.fit(X_train,y_train)

report_model(pruned_tree)

import pickle

# Assuming you already have your trained model 'pruned_tree'
iris_model = 'decision_tree_model.pkl'  # Choose a filename
pickle.dump(pruned_tree, open(iris_model, 'wb'))

import pickle

loaded_model = pickle.load(open('decision_tree_model.pkl', 'rb'))

row_100 = X.iloc[min(199, X.shape[0] - 1)]  # Adjust index to avoid out-of-bounds error
row_100

new_data =[[45,2.3,3.7,5.2,1.7]]
prediction = loaded_model.predict(new_data)
print(prediction)


import streamlit as st
import pickle



def get_Id():
    return st.text_input("Id")
def get_SepalLengthCm():
    SepalLengthCm = st.text_input("Sepal Length")
    return SepalLengthCm

def get_SepalWidthCm():
    SepalWidthCm = st.text_input("sepal width")
    return SepalWidthCm

def get_PetalWidthCm():
    PetalWidthCm = st.text_input("petal width")
    return PetalWidthCm

def get_PetalLengthCm():
    PetalLengthCm = st.text_input("petal length")
    return PetalLengthCm



def predict_species(il,sl, sw, pl, pw):
    try:
        loaded_model = pickle.load(open('decision_tree_model.pkl', 'rb'))
        new_data = np.array([[float(il),float(sl), float(sw), float(pl), float(pw)]])
        prediction = loaded_model.predict(new_data)
        st.write("Prediction: ", prediction[0])
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Streamlit UI
def main():
    st.title('Iris Species Prediction with Decision Tree Model')
    id_value= get_Id()
    sepal_length = get_SepalLengthCm()
    sepal_width = get_SepalWidthCm()
    petal_length = get_PetalLengthCm()
    petal_width = get_PetalWidthCm()
    
    st.write("You entered:")
    st.write(f"Id value: {id_value}")
    st.write(f"Sepal Length: {sepal_length}")
    st.write(f"Sepal Width: {sepal_width}")
    st.write(f"Petal Length: {petal_length}")
    st.write(f"Petal Width: {petal_width}")
    
    if st.button("Predict"):
        predict_species(id_value,sepal_length, sepal_width, petal_length, petal_width)

if __name__ == "__main__":
    main()






