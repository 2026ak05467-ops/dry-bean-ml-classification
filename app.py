
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from sklearn.preprocessing import label_binarize


st.set_page_config(
    page_title="Dry Bean Classification",
    layout="wide"
)

st.title("Dry Bean Classification")
st.write("Machine Learning Assignment 2")

st.write(
    "Upload the test dataset and select a trained classification "
    "model to evaluate its performance."
)


# -----------------------------
# Model selection
# -----------------------------

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib"
}

selected_model = st.selectbox(
    "Select Classification Model",
    list(model_files.keys())
)


# -----------------------------
# Upload CSV
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df.head())

    # Check target column
    if "Class" not in df.columns:

        st.error(
            "The uploaded CSV must contain a 'Class' column."
        )

    else:

        X_test = df.drop("Class", axis=1)
        y_test = df["Class"]

        # Load previously trained model
        model = joblib.load(
            model_files[selected_model]
        )

        # Generate predictions
        y_pred = model.predict(X_test)

        # Generate class probabilities
        y_prob = model.predict_proba(X_test)

        # -----------------------------
        # Evaluation metrics
        # -----------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )

        y_test_bin = label_binarize(
            y_test,
            classes=model.classes_
        )

        auc = roc_auc_score(
            y_test_bin,
            y_prob,
            multi_class="ovr",
            average="weighted"
        )


        # -----------------------------
        # Display metrics
        # -----------------------------

        st.subheader(
            f"{selected_model} Performance"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "AUC",
            f"{auc:.4f}"
        )

        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "MCC",
            f"{mcc:.4f}"
        )


        # -----------------------------
        # Confusion Matrix
        # -----------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=model.classes_
        )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(
            f"{selected_model} Confusion Matrix"
        )

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        st.pyplot(fig)


        # -----------------------------
        # Classification Report
        # -----------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(4)
        )


else:

    st.info(
        "Please upload test_data.csv to evaluate a model."
    )
